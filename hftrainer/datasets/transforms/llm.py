"""Text formatting and tokenization transforms for LLM datasets."""

from __future__ import annotations

from typing import Optional

from hftrainer.registry import TRANSFORMS


@TRANSFORMS.register_module()
class FormatAlpacaPrompt:
    """Format Alpaca instruction data into prompt/full-text fields."""

    PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task"
        "{input_part}. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )
    INPUT_PART = ", paired with an input that provides further context"

    def __call__(self, results: dict) -> dict:
        instruction = results.get('instruction', '')
        context = results.get('input', '')
        output = results.get('output', '')
        input_part = self.INPUT_PART if context else ''
        prompt = self.PROMPT_TEMPLATE.format(
            input_part=input_part,
            instruction=instruction,
        )
        if context:
            prompt = prompt.replace(
                '### Response:\n',
                f'### Input:\n{context}\n\n### Response:\n',
            )
        results['prompt'] = prompt
        results['full_text'] = prompt + output
        return results


@TRANSFORMS.register_module()
class TokenizeAlpacaSample:
    """Tokenize Alpaca prompt/response pairs into model inputs."""

    def __init__(
        self,
        tokenizer_name_or_path: str,
        max_length: int = 512,
        add_eos_token: bool = True,
        padding_side: str = 'right',
    ):
        from transformers import AutoTokenizer

        self.max_length = max_length
        self.add_eos_token = add_eos_token
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = padding_side

    def __call__(self, results: dict) -> dict:
        prompt = results['prompt']
        output = results.get('output', '')
        full_text = results['full_text']
        if self.add_eos_token and self.tokenizer.eos_token and not full_text.endswith(self.tokenizer.eos_token):
            full_text = full_text + self.tokenizer.eos_token

        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        prompt_encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        prompt_len = prompt_encoded['input_ids'].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        results['input_ids'] = input_ids
        results['attention_mask'] = attention_mask
        results['labels'] = labels
        results['input_prompts'] = prompt
        results['output_texts'] = output
        return results
