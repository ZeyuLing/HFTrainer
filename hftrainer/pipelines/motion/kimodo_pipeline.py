"""KIMODO inference pipeline facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from hftrainer.registry import PIPELINES


def _to_tensor(value, *, dtype=torch.float32, device=None):
    if isinstance(value, torch.Tensor):
        out = value
    else:
        out = torch.tensor(value, dtype=dtype)
    if dtype is not None:
        out = out.to(dtype=dtype)
    if device is not None:
        out = out.to(device)
    return out


@PIPELINES.register_module()
class KIMODOPipeline:
    """Unified KIMODO wrapper for text and kinematic-control generation."""

    def __init__(self, bundle):
        self.bundle = bundle

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None, **kwargs):
        """Build a KIMODO pipeline from a bundle config."""
        from hftrainer.models.motion.kimodo import KIMODOBundle

        bundle = KIMODOBundle.from_config(cfg, **kwargs)
        return cls(bundle)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Build a KIMODO pipeline from an hftrainer artifact or KIMODO name."""
        from hftrainer.models.motion.kimodo import KIMODOBundle

        bundle = KIMODOBundle.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(bundle)

    @property
    def skeleton(self):
        return self.bundle.skeleton

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if str(batch.get("task", "")).lower() == "tp2m":
            prompts = batch.get("prompt", batch.get("prompts", batch.get("caption")))
            gt_motions = batch.get("gt_motions_135", batch.get("gt_motion_135"))
            if prompts is None or gt_motions is None:
                raise ValueError("KIMODO TP2M batch needs prompts and gt_motions_135.")
            return {
                "samples": self.infer_tp2m(
                    prompts if isinstance(prompts, (list, tuple)) else [prompts],
                    gt_motions if isinstance(gt_motions, (list, tuple)) else [gt_motions],
                    condition_frames=int(batch.get("condition_frames", batch.get("condition_num_frames", 1))),
                    target_fps=float(batch.get("target_fps", 30.0)),
                    force_clean_prefix=bool(batch.get("force_clean_prefix", True)),
                    force_single_segment=bool(batch.get("force_single_segment", True)),
                    postprocess=batch.get("postprocess"),
                    max_segment_frames=batch.get("max_segment_frames"),
                )
            }
        prompts = batch.get("prompt", batch.get("prompts", batch.get("caption")))
        if prompts is None:
            raise ValueError("KIMODOPipeline batch needs prompt/prompts/caption.")
        num_frames = batch.get("num_frames")
        if num_frames is None:
            duration = batch.get("duration", batch.get("duration_sec"))
            if duration is None:
                raise ValueError("KIMODOPipeline batch needs num_frames or duration.")
            fps = float(getattr(self.bundle.model, "fps", 30))
            num_frames = int(float(duration) * fps)
        constraints = batch.get("constraints")
        return self.bundle.generate(
            prompts=prompts,
            num_frames=num_frames,
            constraints=constraints,
            multi_prompt=bool(batch.get("multi_prompt", False)),
            return_numpy=bool(batch.get("return_numpy", True)),
            **batch.get("generation_kwargs", {}),
        )

    def text_to_motion(self, prompt: str, num_frames: int, **kwargs) -> Dict[str, Any]:
        return self.bundle.generate(prompt, num_frames, constraints=None, **kwargs)

    def multi_prompt(
        self,
        prompts: Sequence[str],
        num_frames: Sequence[int],
        **kwargs,
    ) -> Dict[str, Any]:
        return self.bundle.generate(
            list(prompts),
            list(num_frames),
            constraints=None,
            multi_prompt=True,
            **kwargs,
        )

    def constrained_motion(
        self,
        prompt: str,
        num_frames: int,
        constraints: Sequence[Any],
        **kwargs,
    ) -> Dict[str, Any]:
        return self.bundle.generate(
            prompt,
            num_frames,
            constraints=list(constraints),
            **kwargs,
        )

    def infer_tp2m(
        self,
        prompts: Sequence[str],
        gt_motions_135: Sequence[np.ndarray],
        condition_frames: int,
        target_fps: float = 30.0,
        force_clean_prefix: bool = True,
        force_single_segment: bool = True,
        postprocess: Optional[bool] = None,
        max_segment_frames: Optional[int] = None,
    ) -> list[Dict[str, Any]]:
        """Generate KIMODO TP2M samples from text plus GT Motion135 prefix."""
        if len(prompts) != len(gt_motions_135):
            raise ValueError("prompts and gt_motions_135 must have equal length")
        if int(condition_frames) < 1:
            raise ValueError("condition_frames must be >= 1")

        from scripts.eval.gen_kimodo_tp2m_smplx import _run_one

        model = self.bundle.model
        do_postprocess = self.bundle.post_processing if postprocess is None else bool(postprocess)
        outputs: list[Dict[str, Any]] = []
        for prompt, gt_motion in zip(prompts, gt_motions_135):
            outputs.append(
                _run_one(
                    model,
                    str(prompt),
                    np.asarray(gt_motion, dtype=np.float32),
                    float(target_fps),
                    int(condition_frames),
                    do_postprocess,
                    force_clean_prefix=bool(force_clean_prefix),
                    force_single_segment=bool(force_single_segment),
                    max_segment_frames=max_segment_frames,
                )
            )
        return outputs

    def constraints_from_json(self, path_or_data, *, device=None, dtype=torch.float32):
        self.bundle.load_model()
        from hftrainer.models.motion.kimodo.network.constraints import load_constraints_lst

        return load_constraints_lst(
            str(path_or_data) if isinstance(path_or_data, Path) else path_or_data,
            self.skeleton,
            device=device,
            dtype=dtype,
        )

    def root2d_constraint(
        self,
        frame_indices,
        smooth_root_2d,
        global_root_heading: Optional[Any] = None,
        *,
        device=None,
    ):
        self.bundle.load_model()
        from hftrainer.models.motion.kimodo.network.constraints import Root2DConstraintSet

        return Root2DConstraintSet(
            self.skeleton,
            _to_tensor(frame_indices, dtype=torch.long, device=device),
            _to_tensor(smooth_root_2d, device=device),
            global_root_heading=(
                None
                if global_root_heading is None
                else _to_tensor(global_root_heading, device=device)
            ),
        )

    def fullbody_keyframe_constraint(
        self,
        frame_indices,
        global_joints_positions,
        global_joints_rots,
        smooth_root_2d: Optional[Any] = None,
        *,
        device=None,
    ):
        self.bundle.load_model()
        from hftrainer.models.motion.kimodo.network.constraints import FullBodyConstraintSet

        return FullBodyConstraintSet(
            self.skeleton,
            _to_tensor(frame_indices, dtype=torch.long, device=device),
            _to_tensor(global_joints_positions, device=device),
            _to_tensor(global_joints_rots, device=device),
            smooth_root_2d=(
                None
                if smooth_root_2d is None
                else _to_tensor(smooth_root_2d, device=device)
            ),
        )

    def end_effector_constraint(
        self,
        frame_indices,
        global_joints_positions,
        global_joints_rots,
        smooth_root_2d,
        joint_names: Sequence[str],
        *,
        device=None,
    ):
        self.bundle.load_model()
        from hftrainer.models.motion.kimodo.network.constraints import EndEffectorConstraintSet

        return EndEffectorConstraintSet(
            self.skeleton,
            _to_tensor(frame_indices, dtype=torch.long, device=device),
            _to_tensor(global_joints_positions, device=device),
            _to_tensor(global_joints_rots, device=device),
            _to_tensor(smooth_root_2d, device=device),
            joint_names=list(joint_names),
        )

    def left_hand_constraint(self, *args, **kwargs):
        return self._named_end_effector_constraint("LeftHandConstraintSet", *args, **kwargs)

    def right_hand_constraint(self, *args, **kwargs):
        return self._named_end_effector_constraint("RightHandConstraintSet", *args, **kwargs)

    def left_foot_constraint(self, *args, **kwargs):
        return self._named_end_effector_constraint("LeftFootConstraintSet", *args, **kwargs)

    def right_foot_constraint(self, *args, **kwargs):
        return self._named_end_effector_constraint("RightFootConstraintSet", *args, **kwargs)

    def _named_end_effector_constraint(
        self,
        class_name: str,
        frame_indices,
        global_joints_positions,
        global_joints_rots,
        smooth_root_2d,
        *,
        device=None,
    ):
        self.bundle.load_model()
        from hftrainer.models.motion.kimodo.network import constraints

        cls = getattr(constraints, class_name)
        return cls(
            self.skeleton,
            _to_tensor(frame_indices, dtype=torch.long, device=device),
            _to_tensor(global_joints_positions, device=device),
            _to_tensor(global_joints_rots, device=device),
            _to_tensor(smooth_root_2d, device=device),
        )
