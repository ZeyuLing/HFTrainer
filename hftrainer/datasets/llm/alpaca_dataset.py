"""Alpaca-format instruction tuning dataset."""

import json
import os
from typing import Dict, Any, Optional, List

import torch

from hftrainer.datasets.llm.base_llm_dataset import BaseLLMDataset
from hftrainer.registry import DATASETS


@DATASETS.register_module()
class AlpacaDataset(BaseLLMDataset):
    """
    Dataset for Alpaca-format instruction tuning data.

    Expected JSON format (list of dicts):
        [
            {
                "instruction": "What is the capital of France?",
                "input": "",           # optional context
                "output": "Paris"
            },
            ...
        ]

    Config example:
        dataset=dict(
            type='AlpacaDataset',
            data_path='data/llm/demo/alpaca_sample.json',
            tokenizer_name_or_path='checkpoints/TinyLlama-1.1B-Chat-v1.0',
            max_length=512,
        )
    """

    PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task"
        "{input_part}. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )
    INPUT_PART = ", paired with an input that provides further context"

    def __init__(
        self,
        data_path: str,
        tokenizer_name_or_path: str,
        max_length: int = 512,
        split: str = 'train',
        max_samples: Optional[int] = None,
    ):
        self.max_length = max_length

        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'right'

        # Load data
        if os.path.isfile(data_path):
            with open(data_path) as f:
                data = json.load(f)
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")

        if not isinstance(data, list):
            raise ValueError(f"Expected list of dicts, got {type(data)}")

        self.data = data
        if max_samples is not None:
            self.data = self.data[:max_samples]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        instruction = item.get('instruction', '')
        context = item.get('input', '')
        output = item.get('output', '')

        # Format prompt
        input_part = self.INPUT_PART if context else ''
        prompt = self.PROMPT_TEMPLATE.format(
            input_part=input_part,
            instruction=instruction,
        )
        if context:
            # Insert context before Response
            prompt = prompt.replace(
                '### Response:\n',
                f'### Input:\n{context}\n\n### Response:\n'
            )

        # Full text = prompt + response
        full_text = prompt + output + self.tokenizer.eos_token

        # Tokenize
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        # Create labels: mask prompt tokens with -100
        prompt_encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        prompt_len = prompt_encoded['input_ids'].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # mask prompt
        labels[attention_mask == 0] = -100  # mask padding

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'input_prompts': prompt,
            'output_texts': output,
        }
