"""PRISM inference pipeline wrapper."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from hftrainer.pipelines.base_pipeline import BasePipeline
from hftrainer.registry import PIPELINES


@PIPELINES.register_module()
class PrismPipeline(BasePipeline):
    """HFTrainer wrapper around the vendored PRISM AR pipeline."""

    def __init__(self, bundle, **kwargs):
        super().__init__(bundle)
        from hftrainer.pipelines.motion.prism_backend import PrismARPipeline

        self.backend = PrismARPipeline(
            tokenizer=bundle.tokenizer,
            text_encoder=bundle.text_encoder,
            vae=bundle.vae,
            scheduler=bundle.scheduler,
            smpl_processor=bundle.smpl_pose_processor,
            transformer=bundle.transformer,
        )

    def __call__(
        self,
        prompts: Union[str, List[str]],
        negative_prompt: Optional[str] = None,
        first_frame_motion_path: Optional[str] = None,
        num_frames_per_segment: Union[int, List[int]] = 129,
        num_joints: int = 23,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        **kwargs,
    ) -> Dict[str, Any]:
        return self.backend(
            prompts=prompts,
            negative_prompt=negative_prompt,
            first_frame_motion_path=first_frame_motion_path,
            num_frames_per_segment=num_frames_per_segment,
            num_joints=num_joints,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        )
