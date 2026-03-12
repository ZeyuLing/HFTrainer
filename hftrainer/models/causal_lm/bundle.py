"""CausalLM ModelBundle for LLM training."""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.registry import MODEL_BUNDLES


@MODEL_BUNDLES.register_module()
class CausalLMBundle(ModelBundle):
    """
    ModelBundle for causal language model (LLaMA, Qwen, Mistral, etc.) SFT.

    Sub-modules:
      - model: AutoModelForCausalLM (trainable or LoRA)
      - tokenizer: AutoTokenizer (not nn.Module, stored as plain attribute)

    Atomic forward functions:
      - tokenize(texts) -> {input_ids, attention_mask}
      - forward_logits(input_ids, attention_mask, labels) -> CausalLMOutput
      - generate(prompts, **kwargs) -> list[str]
    """

    def __init__(
        self,
        model: dict,
        tokenizer_path: Optional[str] = None,
        max_length: int = 512,
        padding_side: str = 'right',
    ):
        super().__init__()
        self.max_length = max_length
        self.padding_side = padding_side

        # Build sub-modules (model only; tokenizer is handled separately)
        self._build_modules({'model': model})

        # Load tokenizer
        pretrained_path = tokenizer_path
        if pretrained_path is None:
            model_cfg = model if isinstance(model, dict) else {}
            fp = model_cfg.get('from_pretrained', {})
            pretrained_path = fp.get('pretrained_model_name_or_path') if fp else None

        if pretrained_path is not None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(pretrained_path)
            # LLaMA family: no pad token by default
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
                tok.pad_token_id = tok.eos_token_id
            tok.padding_side = padding_side
            self.tokenizer = tok
        else:
            self.tokenizer = None

    @classmethod
    def _bundle_config_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        model_type: str = 'AutoModelForCausalLM',
        model_kwargs: Optional[Dict[str, Any]] = None,
        model_overrides: Optional[Dict[str, Any]] = None,
        tokenizer_path: Optional[str] = None,
        max_length: int = 512,
        padding_side: str = 'right',
    ) -> Dict[str, Any]:
        model_cfg = {
            'type': model_type,
            'from_pretrained': {
                'pretrained_model_name_or_path': pretrained_model_name_or_path,
            },
        }
        cls._merge_nested_dict(model_cfg['from_pretrained'], model_kwargs)
        cls._merge_nested_dict(model_cfg, model_overrides)
        return {
            'model': model_cfg,
            'tokenizer_path': tokenizer_path or pretrained_model_name_or_path,
            'max_length': max_length,
            'padding_side': padding_side,
        }

    def save_pretrained(
        self,
        save_directory: str,
        merge_lora: bool = True,
        safe_serialization: bool = True,
        **kwargs,
    ):
        from hftrainer.utils.hf_export import safe_hf_export

        os.makedirs(save_directory, exist_ok=True)
        if merge_lora and self.is_lora_module('model'):
            self.merge_lora_weights(['model'])

        with safe_hf_export():
            self.model.save_pretrained(
                save_directory,
                safe_serialization=safe_serialization,
                **kwargs,
            )
        if self.tokenizer is not None and hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(save_directory)

    def tokenize(
        self,
        texts: List[str],
        labels_texts: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts.

        Args:
            texts: list of input text strings
            labels_texts: if provided, tokenize as labels (mask prompt tokens)

        Returns:
            dict with 'input_ids', 'attention_mask', optionally 'labels'
        """
        assert self.tokenizer is not None, "Tokenizer not initialized"

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

        if labels_texts is not None:
            # Create labels: mask padding tokens with -100
            labels = encoded['input_ids'].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            result['labels'] = labels
        else:
            # Use input_ids as labels (standard LM objective)
            labels = encoded['input_ids'].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            result['labels'] = labels

        return result

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Run forward pass.

        Args:
            input_ids: Tensor[B, L]
            attention_mask: Tensor[B, L]
            labels: Tensor[B, L] with -100 for masked positions

        Returns:
            CausalLMOutputWithPast (or similar) with .loss and .logits
        """
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
        """
        Generate text completions.

        Args:
            prompts: list of prompt strings
            max_new_tokens: maximum number of new tokens to generate
            temperature: sampling temperature
            do_sample: whether to sample (True) or use greedy decoding (False)
            **kwargs: passed to model.generate()

        Returns:
            list of generated text strings (prompt + completion)
        """
        assert self.tokenizer is not None, "Tokenizer not initialized"

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
                **kwargs,
            )

        # Decode only the new tokens
        generated_texts = []
        for i, out_ids in enumerate(output_ids):
            new_tokens = out_ids[input_ids.shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts
