"""Alpaca-format instruction tuning dataset."""

import json
import os
from typing import Optional

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
        pipeline=None,
        serialize_data: bool = False,
    ):
        self.data_path = data_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.max_length = max_length
        self.split = split
        self.max_samples = max_samples
        super().__init__(pipeline=pipeline, serialize_data=serialize_data)

    def build_default_pipeline(self):
        return [
            dict(type='FormatAlpacaPrompt'),
            dict(
                type='TokenizeAlpacaSample',
                tokenizer_name_or_path=self.tokenizer_name_or_path,
                max_length=self.max_length,
            ),
        ]

    def load_data_list(self):
        if not os.path.isfile(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        with open(self.data_path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected list of dicts, got {type(data)}")
        if self.max_samples is not None:
            data = data[:self.max_samples]
        return data
