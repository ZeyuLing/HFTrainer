"""VerMo multimodal-token bundle."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.registry import MODEL_BUNDLES


@MODEL_BUNDLES.register_module()
class VermoBundle(ModelBundle):
    """ModelBundle for VerMo multimodal motion generation."""

    def __init__(
        self,
        processor: dict,
        lm: dict,
        mean_init_embeddings: bool = False,
    ):
        super().__init__()
        self.mean_init_embeddings = mean_init_embeddings
        self._build_modules({'processor': processor, 'lm': lm})
        self._resize_token_embeddings()

    @classmethod
    def _bundle_config_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
    ) -> Dict[str, Any]:
        root = os.path.abspath(os.path.expanduser(pretrained_model_name_or_path))
        cfg_path = os.path.join(root, 'bundle_config.json')
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(
                f"Expected bundle_config.json under {root}. "
                "Use VermoBundle.save_pretrained() exports, or construct via from_config()."
            )
        with open(cfg_path, 'r', encoding='utf-8') as f:
            bundle_cfg = json.load(f)

        processor_cfg = bundle_cfg['processor']
        lm_cfg = bundle_cfg['lm']
        text_tok_cfg = processor_cfg.get('pretrained_text_tokenizer', {})
        fp = text_tok_cfg.get('from_pretrained') or {}
        fp['pretrained_model_name_or_path'] = os.path.join(root, 'tokenizer')
        text_tok_cfg['from_pretrained'] = fp
        lm_fp = lm_cfg.get('from_pretrained') or {}
        lm_fp['pretrained_model_name_or_path'] = os.path.join(root, 'lm')
        lm_cfg['from_pretrained'] = lm_fp

        return {
            'processor': processor_cfg,
            'lm': lm_cfg,
            'mean_init_embeddings': bundle_cfg.get('mean_init_embeddings', False),
        }

    def _resize_token_embeddings(self):
        if hasattr(self.lm, 'resize_token_embeddings'):
            self.lm.resize_token_embeddings(
                self.processor.vocab_size,
                mean_resizing=self.mean_init_embeddings,
            )

    def save_pretrained(self, save_directory: str, merge_lora: bool = True, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        if merge_lora and self.is_lora_module('lm'):
            self.merge_lora_weights(['lm'])

        self.lm.save_pretrained(os.path.join(save_directory, 'lm'), **kwargs)
        self.processor.text_tokenizer.save_pretrained(os.path.join(save_directory, 'tokenizer'))
        with open(os.path.join(save_directory, 'bundle_config.json'), 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'processor': self.get_module_build_cfg('processor'),
                    'lm': self.get_module_build_cfg('lm'),
                    'mean_init_embeddings': self.mean_init_embeddings,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def process_train(self, inputs: Dict[str, Any]):
        return self.processor.process_train(inputs)

    def forward_lm(self, inputs: Dict[str, Any]):
        return self.lm(**self.process_train(inputs))
