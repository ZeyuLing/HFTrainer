"""Causal language-model bundle backed by local LLaMA and tokenizer code."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.models.llama.network import LocalLlamaForCausalLM
from hftrainer.tokenization import LocalTokenizer
from hftrainer.registry import MODEL_BUNDLES


@MODEL_BUNDLES.register_module()
class LlamaBundle(ModelBundle):
    """Shared training/inference boundary for repository-local causal LMs."""

    PRETRAINED_SPEC = {
        'components': {
            'model': {
                'default_type': 'LocalLlamaForCausalLM',
                'type_arg': 'model_type',
                'pretrained_kwargs_arg': 'model_kwargs',
                'overrides_arg': 'model_overrides',
            },
        },
        'init_args': {
            'tokenizer_path': {'default': ModelBundle._PRETRAINED_PATH_SENTINEL},
            'max_length': 512,
            'padding_side': 'right',
        },
    }
    def __init__(
        self,
        model: dict | LocalLlamaForCausalLM,
        tokenizer_path: Optional[str] = None,
        tokenizer: Optional[LocalTokenizer] = None,
        max_length: int = 512,
        padding_side: str = 'right',
    ):
        super().__init__()
        self.max_length = int(max_length)
        self.padding_side = padding_side
        if tokenizer_path is None and isinstance(model, dict):
            pretrained = model.get('from_pretrained') or {}
            tokenizer_path = pretrained.get('pretrained_model_name_or_path')
        self._build_modules({'model': model})
        if type(self.model) is not LocalLlamaForCausalLM:
            raise TypeError(
                'LlamaBundle.model must be LocalLlamaForCausalLM; '
                f'got {type(self.model).__module__}.{type(self.model).__name__}.'
            )
        if tokenizer is not None and tokenizer_path is not None:
            raise ValueError('Provide tokenizer or tokenizer_path, not both.')
        if tokenizer is not None and type(tokenizer) is not LocalTokenizer:
            raise TypeError('tokenizer must be an hftrainer LocalTokenizer.')
        if tokenizer is None and tokenizer_path is not None:
            tokenizer = LocalTokenizer.from_pretrained(
                tokenizer_path, padding_side=padding_side
            )
        if tokenizer is not None:
            tokenizer.padding_side = padding_side
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is None:
                    raise ValueError('Tokenizer needs pad_token or eos_token for batching.')
                tokenizer.pad_token = tokenizer.eos_token
            if len(tokenizer) > self.model.config.vocab_size:
                raise ValueError(
                    f'Tokenizer IDs require vocab size {len(tokenizer)}, but model has '
                    f'{self.model.config.vocab_size} embeddings.'
                )
        self.tokenizer = tokenizer

    def tokenize(
        self,
        texts: List[str],
        labels_texts: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError('Tokenizer is not initialized.')
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        result = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
        }
        if labels_texts is None:
            labels = encoded['input_ids'].clone()
        else:
            label_batch = self.tokenizer(
                labels_texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            )
            labels = label_batch['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        result['labels'] = labels
        return result

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        do_sample: bool = False,
        **kwargs,
    ) -> List[str]:
        if self.tokenizer is None:
            raise RuntimeError('Tokenizer is not initialized.')
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max(1, self.max_length - max_new_tokens),
        )
        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs['attention_mask'].to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
        prompt_width = input_ids.shape[1]
        return [
            self.tokenizer.decode(ids[prompt_width:], skip_special_tokens=True)
            for ids in output_ids
        ]

    def save_pretrained(
        self,
        save_directory: str,
        merge_lora: bool = True,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        if merge_lora and self.is_lora_module('model'):
            self.merge_lora_weights(['model'])
        self.model.save_pretrained(
            save_directory, safe_serialization=safe_serialization, **kwargs
        )
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)
