"""Plain-text causal language modelling without instruction formatting."""

import json
from pathlib import Path

import torch

from hftrainer.datasets.instruction_sft.base import BaseLLMDataset
from hftrainer.registry import DATASETS
from hftrainer.tokenization import LocalTokenizer


@DATASETS.register_module()
class PlainTextDataset(BaseLLMDataset):
    """JSONL text rows, independently truncated with padding excluded from loss.

    Each example retains its own boundary; examples are never concatenated
    across splits. EOS is supervised when there is room after truncation.
    """

    def __init__(
        self,
        data_path,
        tokenizer_name_or_path,
        max_length=256,
        max_samples=None,
        pipeline=None,
        serialize_data=False,
    ):
        if max_length < 2:
            raise ValueError("max_length must be at least 2 for next-token prediction")
        self.data_path = data_path
        self.max_length = max_length
        self.max_samples = max_samples
        self.tokenizer = LocalTokenizer.from_pretrained(tokenizer_name_or_path)
        super().__init__(pipeline=pipeline, serialize_data=serialize_data)

    def build_default_pipeline(self):
        return [self.tokenize]

    def load_data_list(self):
        records = [
            json.loads(line)
            for line in Path(self.data_path).read_text().splitlines()
            if line.strip()
        ]
        records = [r for r in records if r["text"].strip()]
        return records if self.max_samples is None else records[: self.max_samples]

    def tokenize(self, results):
        ids = self.tokenizer.encode(results["text"], add_special_tokens=True)[
            : self.max_length
        ]
        eos = self.tokenizer.eos_token_id
        if (
            eos is not None
            and len(ids) < self.max_length
            and (not ids or ids[-1] != eos)
        ):
            ids.append(eos)
        if len(ids) < 2:
            raise ValueError("Plain-text example has fewer than two tokens")
        length = len(ids)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = eos if eos is not None else 0
        input_ids = torch.tensor(
            ids + [pad] * (self.max_length - length), dtype=torch.long
        )
        attention_mask = torch.arange(self.max_length) < length
        labels = input_ids.clone()
        labels[~attention_mask] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask.long(),
            "labels": labels,
        }
