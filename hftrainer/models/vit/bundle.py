"""ViT classification ModelBundle."""

import torch
from typing import Optional, Tuple

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.utils.image import IMAGENET_MEAN, IMAGENET_STD, normalize_image, pil_to_tensor


@MODEL_BUNDLES.register_module()
class ViTBundle(ModelBundle):
    """
    ModelBundle for ViT-based image classification.

    Sub-modules:
      - model: ViTForImageClassification (trainable)

    Atomic forward functions shared by Trainer and Pipeline:
      - preprocess(images) → pixel_values
      - forward_features(pixel_values) → logits
      - classify(pixel_values) → (pred_ids, scores)
    """

    HF_PRETRAINED_SPEC = {
        'components': {
            'model': {
                'default_type': 'AutoModelForImageClassification',
                'type_arg': 'model_type',
                'pretrained_kwargs_arg': 'model_kwargs',
                'overrides_arg': 'model_overrides',
            },
        },
        'init_args': {
            'num_labels': None,
            'image_size': 224,
        },
    }
    HF_SAVE_PRETRAINED_SPEC = {
        'kind': 'module',
        'module': 'model',
        'merge_lora_modules': ['model'],
        'extra_artifacts': ['_image_processor'],
    }

    def __init__(
        self,
        model: dict,
        num_labels: Optional[int] = None,
        image_size: int = 224,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_labels = num_labels

        # Build sub-modules
        self._build_modules({'model': model})

        # Load feature extractor / image processor for preprocessing
        pretrained_path = None
        if hasattr(model, 'get'):
            fp = model.get('from_pretrained', {})
            pretrained_path = fp.get('pretrained_model_name_or_path') if fp else None

        self._image_processor = None
        if pretrained_path:
            try:
                from transformers import AutoImageProcessor
                self._image_processor = AutoImageProcessor.from_pretrained(pretrained_path)
            except Exception:
                pass

    @classmethod
    def _bundle_config_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        num_labels: Optional[int] = None,
        **kwargs,
    ):
        bundle_cfg = cls._build_bundle_config_from_spec(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            spec=cls.HF_PRETRAINED_SPEC,
            num_labels=num_labels,
            **kwargs,
        )
        if num_labels is not None:
            bundle_cfg['model']['from_pretrained']['num_labels'] = num_labels
            bundle_cfg['model']['from_pretrained'].setdefault(
                'ignore_mismatched_sizes',
                True,
            )
        return bundle_cfg

    def preprocess(self, images) -> torch.Tensor:
        """
        Preprocess images to pixel_values tensor.

        Args:
            images: list of PIL Images, or Tensor[B, C, H, W] in [0,1]

        Returns:
            pixel_values: Tensor[B, 3, H, W] normalized
        """
        if isinstance(images, torch.Tensor):
            return images  # assume already preprocessed

        if self._image_processor is not None:
            inputs = self._image_processor(images=images, return_tensors='pt')
            return inputs['pixel_values']

        # Fallback: manual normalization
        tensors = [pil_to_tensor(img) for img in images]
        pixel_values = torch.stack(tensors)
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return normalize_image(pixel_values, mean.flatten().tolist(), std.flatten().tolist())

    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Run ViT forward pass and return logits.

        Args:
            pixel_values: Tensor[B, 3, H, W]

        Returns:
            logits: Tensor[B, num_classes]
        """
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    def classify(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify images.

        Args:
            pixel_values: Tensor[B, 3, H, W]

        Returns:
            pred_ids: Tensor[B] — predicted class indices
            scores: Tensor[B, num_classes] — softmax probabilities
        """
        logits = self.forward_features(pixel_values)
        scores = torch.softmax(logits, dim=-1)
        pred_ids = scores.argmax(dim=-1)
        return pred_ids, scores
