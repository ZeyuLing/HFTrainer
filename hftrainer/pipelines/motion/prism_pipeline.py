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

        # Forward backend-init options from config kwargs.
        backend_init_keys = ("expand_timesteps", "is_causal", "dtype")
        backend_kwargs = {k: kwargs[k] for k in backend_init_keys if k in kwargs}

        self.backend = PrismARPipeline(
            tokenizer=getattr(bundle, "tokenizer", None),
            text_encoder=getattr(bundle, "text_encoder", None),
            vae=bundle.vae,
            scheduler=bundle.scheduler,
            smpl_processor=bundle.smpl_pose_processor,
            transformer=bundle.transformer,
            **backend_kwargs,
        )

    @staticmethod
    def _max_frames(num_frames_per_segment: Union[int, List[int]]) -> int:
        if isinstance(num_frames_per_segment, list):
            if not num_frames_per_segment:
                return 0
            return max(int(n) for n in num_frames_per_segment)
        return int(num_frames_per_segment)

    @staticmethod
    def _pad_canvas(
        num_frames_per_segment: Union[int, List[int]],
        pad_to_frames: int,
    ) -> Union[int, List[int]]:
        if isinstance(num_frames_per_segment, list):
            return [int(pad_to_frames)] * len(num_frames_per_segment)
        return int(pad_to_frames)

    def _length_policy_kwargs(
        self,
        length_policy: Optional[str],
        num_frames_per_segment: Union[int, List[int]],
        pad_to_frames: int,
        strict_length: bool,
    ) -> Dict[str, Any]:
        """Translate public length policy into backend generation kwargs."""
        if length_policy in (None, "legacy"):
            return {}

        if length_policy == "direct_len":
            return {
                "generation_num_frames_per_segment": num_frames_per_segment,
                "valid_num_frames_per_segment": num_frames_per_segment,
                "preserve_segment_lengths": True,
                "allow_segment_padding": not strict_length,
                "align_generation_frames": True,
            }

        if length_policy == "pad360_crop":
            pad_to_frames = int(pad_to_frames)
            if self._max_frames(num_frames_per_segment) > pad_to_frames:
                raise ValueError(
                    "pad360_crop requires every requested segment length to be "
                    f"<= pad_to_frames ({pad_to_frames}); got "
                    f"{num_frames_per_segment!r}"
                )
            return {
                "generation_num_frames_per_segment": self._pad_canvas(
                    num_frames_per_segment, pad_to_frames
                ),
                "valid_num_frames_per_segment": num_frames_per_segment,
                "preserve_segment_lengths": True,
                "allow_segment_padding": not strict_length,
                "align_generation_frames": False,
            }

        raise ValueError(
            "length_policy must be one of direct_len, pad360_crop, legacy, or None; "
            f"got {length_policy!r}"
        )

    def __call__(
        self,
        prompts: Union[str, List[str]],
        negative_prompt: Optional[str] = None,
        first_frame_motion_path: Optional[str] = None,
        condition_num_frames: int = 1,
        num_frames_per_segment: Union[int, List[int]] = 129,
        num_joints: int = 23,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        length_policy: Optional[str] = "pad360_crop",
        pad_to_frames: int = 360,
        strict_length: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        backend_kwargs = dict(kwargs)
        backend_kwargs.setdefault("use_rollout_trans", "xz_rollout_y_absolute")
        for key, value in self._length_policy_kwargs(
            length_policy=length_policy,
            num_frames_per_segment=num_frames_per_segment,
            pad_to_frames=pad_to_frames,
            strict_length=strict_length,
        ).items():
            backend_kwargs.setdefault(key, value)

        return self.backend(
            prompts=prompts,
            negative_prompt=negative_prompt,
            first_frame_motion_path=first_frame_motion_path,
            condition_num_frames=condition_num_frames,
            num_frames_per_segment=num_frames_per_segment,
            num_joints=num_joints,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **backend_kwargs,
        )

    def infer(self, *args, **kwargs) -> Dict[str, Any]:
        """Public PRISM inference entrypoint.

        The default length policy is ``pad360_crop``: generate on the same
        360-frame training canvas, mask tokens beyond the requested valid
        length, then crop the decoded motion back to the requested length.
        The default translation decode is ``xz_rollout_y_absolute``: root x/z
        use rollout from relative channels, while root y uses the decoded
        absolute channel to reduce height drift.
        Pass ``length_policy="direct_len"`` only for exact-length ablations, or
        ``length_policy="legacy"`` for old behavior.
        """
        return self.__call__(*args, **kwargs)
