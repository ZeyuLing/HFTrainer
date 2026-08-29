"""ViT classification bundle backed only by repository-local components."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.models.vit.network import LocalViTForImageClassification
from hftrainer.models.vit.processing import ViTImageProcessor
from hftrainer.registry import MODEL_BUNDLES


@MODEL_BUNDLES.register_module()
class ViTBundle(ModelBundle):
    """Training/inference boundary for the local ViT implementation."""

    PRETRAINED_SPEC = {
        'components': {
            'model': {
                'default_type': 'LocalViTForImageClassification',
                'type_arg': 'model_type',
                'pretrained_kwargs_arg': 'model_kwargs',
                'overrides_arg': 'model_overrides',
            },
        },
        'init_args': {'num_labels': None, 'image_size': 224},
    }
    def __init__(
        self,
        model: dict | LocalViTForImageClassification,
        num_labels: Optional[int] = None,
        image_size: int = 224,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.num_labels = num_labels
        pretrained_path = None
        if isinstance(model, dict):
            from_pretrained = model.get('from_pretrained') or {}
            pretrained_path = from_pretrained.get('pretrained_model_name_or_path')
            if num_labels is not None and from_pretrained:
                model = dict(model)
                model['from_pretrained'] = dict(from_pretrained)
                model['from_pretrained'].setdefault('num_labels', num_labels)
                model['from_pretrained'].setdefault('ignore_mismatched_sizes', True)
        self._build_modules({'model': model})
        if type(self.model) is not LocalViTForImageClassification:
            raise TypeError(
                'ViTBundle.model must be LocalViTForImageClassification; '
                f'got {type(self.model).__module__}.{type(self.model).__name__}.'
            )
        self._image_processor = (
            ViTImageProcessor.from_pretrained(pretrained_path, self.image_size)
            if pretrained_path else ViTImageProcessor(size=self.image_size)
        )

    def preprocess(self, images) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            return images
        return self._image_processor(images=images, return_tensors='pt')['pixel_values']

    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits

    def classify(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward_features(pixel_values)
        scores = torch.softmax(logits, dim=-1)
        return scores.argmax(dim=-1), scores

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
        self._image_processor.save_pretrained(save_directory)
