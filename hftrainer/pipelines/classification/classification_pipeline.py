"""Classification pipeline."""

import torch
from typing import Any, Dict, List, Optional, Union

from hftrainer.pipelines.base_pipeline import BasePipeline
from hftrainer.registry import PIPELINES


@PIPELINES.register_module()
class ClassificationPipeline(BasePipeline):
    """
    Inference pipeline for image classification.

    Usage:
        pipeline = ClassificationPipeline.from_checkpoint(bundle_cfg, ckpt_path)
        results = pipeline(images)
    """

    @torch.no_grad()
    def __call__(
        self,
        images,
        return_scores: bool = False,
    ) -> Union[List[int], Dict[str, Any]]:
        """
        Classify images.

        Args:
            images: single PIL Image, list of PIL Images, or Tensor[B,3,H,W]
            return_scores: if True, also return confidence scores

        Returns:
            list of predicted class ids (or dict with 'preds' and 'scores')
        """
        if isinstance(images, torch.Tensor):
            if images.ndim == 3:
                pixel_values = images.unsqueeze(0)
                single = True
            elif images.ndim == 4:
                pixel_values = images
                single = images.shape[0] == 1
            else:
                raise ValueError(
                    f"Expected image tensor with 3 or 4 dims, got shape {tuple(images.shape)}"
                )
        else:
            if not isinstance(images, (list, tuple)):
                images = [images]
                single = True
            else:
                single = False
            pixel_values = self.bundle.preprocess(images)

        pixel_values = pixel_values.to(next(iter(self.bundle.model.parameters())).device)

        pred_ids, scores = self.bundle.classify(pixel_values)

        pred_ids = pred_ids.cpu().tolist()
        if single:
            pred_ids = pred_ids[0]

        if return_scores:
            return {'preds': pred_ids, 'scores': scores.cpu()}
        return pred_ids
