"""
Unified motion quality checker that composes all individual checkers.

Provides a single entry point to run all quality checks on a motion and
produce an aggregated verdict. Shares body model across FK-based checkers.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from ._geometry_compat import angle_axis_to_rotation_matrix

from .base_checker import BaseQualityChecker, CheckResult, normalize_betas_array, normalize_poses_array
from .mask_utils import build_invalid_mask

# Import all individual checkers
from .jitter_checker import JitterChecker
from .joint_twist_checker import JointTwistChecker
from .candy_wrapper_checker import CandyWrapperChecker
from .joint_jump_checker import JointJumpChecker
from .arm_penetration_checker import ArmPenetrationChecker
from .small_wobble_checker import SmallWobbleChecker
from .foot_sliding_checker import FootSlidingChecker
from .rotation_velocity_checker import RotationVelocityChecker
from .translation_velocity_checker import TranslationVelocityChecker
from .rotation_validity_checker import RotationValidityChecker
from .ported_hymotion_data_checkers import (
    AnkleXChecker,
    FirstFrameRotationVelocityChecker,
    KneeXChecker,
    NeckChecker,
    Spine1Checker,
    Spine2Checker,
    SpineChecker,
)
from .root_motion_utils import root_rotation_matrices_from_poses

from ._model_compat import SmplxLiteJ24, batch_rigid_transform_v2


def _compute_shared_fk_joints_22(
    data: Dict[str, Any],
    body_model: Any,
    device: str,
    poses_3d: Optional[np.ndarray] = None,
) -> np.ndarray:
    trans = np.asarray(data["trans"])
    poses_3d = poses_3d if poses_3d is not None else normalize_poses_array(np.asarray(data["poses"]), num_joints=22)
    betas = normalize_betas_array(data.get("betas"))

    F = poses_3d.shape[0]
    poses_t = torch.as_tensor(poses_3d, dtype=torch.float32, device=device)
    trans_t = torch.as_tensor(trans, dtype=torch.float32, device=device)
    if betas is None:
        betas_t = torch.zeros((1, 16), dtype=torch.float32, device=device)
    else:
        betas_t = torch.as_tensor(betas, dtype=torch.float32, device=device)
        if betas_t.ndim == 1:
            betas_t = betas_t.unsqueeze(0)

    with torch.no_grad():
        if batch_rigid_transform_v2 is not None and hasattr(body_model, "get_skeleton") and hasattr(body_model, "parents"):
            # Checker paths only need joint positions. Avoid full pose blend + skinning.
            rest_joints = body_model.get_skeleton(betas_t)[..., :22, :]
            rot_mats = angle_axis_to_rotation_matrix(poses_t)
            posed_joints, _ = batch_rigid_transform_v2(rot_mats, rest_joints, body_model.parents[:22])
            joints = posed_joints + trans_t[:, None, :]
        else:
            global_orient = poses_t[:, 0, :]
            body_pose = poses_t[:, 1:22, :].reshape(F, 63)
            joints = body_model(
                body_pose=body_pose,
                betas=betas_t,
                global_orient=global_orient,
                transl=trans_t,
                rotation_mode="aa",
            )
    return joints[:, :22, :].detach().cpu().numpy()


class AggregatedCheckResult:
    """Result of running all quality checks on a single motion."""

    def __init__(
        self,
        is_valid: bool,
        failed_checks: List[str],
        borderline_checks: List[str],
        all_results: Dict[str, CheckResult],
        combined_reason: str,
        profiling: Optional[Dict[str, float]] = None,
    ):
        self.is_valid = is_valid
        self.failed_checks = failed_checks
        self.borderline_checks = borderline_checks
        self.all_results = all_results
        self.combined_reason = combined_reason
        self.profiling = dict(profiling or {})
        if self.failed_checks:
            self.category = "low"
        elif self.borderline_checks:
            self.category = "borderline"
        else:
            self.category = "high"

    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "category": self.category,
            "failed_checks": self.failed_checks,
            "borderline_checks": self.borderline_checks,
            "combined_reason": self.combined_reason,
            "per_checker": {
                name: {
                    "is_valid": r.get("is_valid", True),
                    "severity": r.get("severity", "fail" if not r.get("is_valid", True) else "pass"),
                    "invalid_reason": r.get("invalid_reason", ""),
                }
                for name, r in self.all_results.items()
            },
            "profiling": self.profiling,
        }


# All available checker classes and their categories
CHECKER_REGISTRY = {
    "jitter": {"cls": JitterChecker, "needs_fk": True},
    "joint_twist": {"cls": JointTwistChecker, "needs_fk": False},
    "candy_wrapper": {"cls": CandyWrapperChecker, "needs_fk": False},
    "joint_jump": {"cls": JointJumpChecker, "needs_fk": True},
    "arm_penetration": {"cls": ArmPenetrationChecker, "needs_fk": True},
    "small_wobble": {"cls": SmallWobbleChecker, "needs_fk": True},
    "foot_sliding": {"cls": FootSlidingChecker, "needs_fk": True},
    "rotation_velocity": {"cls": RotationVelocityChecker, "needs_fk": False},
    "first_frame_rotation_velocity": {"cls": FirstFrameRotationVelocityChecker, "needs_fk": False},
    "translation_velocity": {"cls": TranslationVelocityChecker, "needs_fk": False},
    "rotation_validity": {"cls": RotationValidityChecker, "needs_fk": False},
    "knee_x": {"cls": KneeXChecker, "needs_fk": False},
    "ankle_x": {"cls": AnkleXChecker, "needs_fk": False},
    "neck": {"cls": NeckChecker, "needs_fk": False},
    "spine": {"cls": SpineChecker, "needs_fk": False},
    "spine1": {"cls": Spine1Checker, "needs_fk": False},
    "spine2": {"cls": Spine2Checker, "needs_fk": False},
}


class MotionQualityChecker:
    """Unified checker that composes all individual checkers.

    Shares a single body model instance across all FK-based checkers to avoid
    loading multiple copies.

    Usage:
        checker = MotionQualityChecker(device="cuda")
        result = checker.check({"poses": ..., "trans": ...})
        result = checker.check_from_file("path/to/motion.npz")
    """

    def __init__(
        self,
        device: str = "cuda",
        enabled_checkers: Optional[List[str]] = None,
        checker_kwargs: Optional[Dict[str, Dict]] = None,
    ):
        """Initialize the unified checker.

        Args:
            device: Device for body model and tensor ops.
            enabled_checkers: List of checker names to enable. If None, all are enabled.
            checker_kwargs: Per-checker kwargs overrides, e.g.
                {"jitter": {"...": ...}, "joint_jump": {"jump_threshold_m": 0.2}}.
        """
        self.device = device
        self.checker_kwargs = checker_kwargs or {}

        # Determine which checkers to enable
        if enabled_checkers is not None:
            self.enabled = [name for name in enabled_checkers if name in CHECKER_REGISTRY]
        else:
            self.enabled = list(CHECKER_REGISTRY.keys())

        # Create shared body model for FK-based checkers
        self.body_model: Optional[Any] = None
        needs_fk = any(CHECKER_REGISTRY[name]["needs_fk"] for name in self.enabled)
        if needs_fk and SmplxLiteJ24 is not None:
            self.body_model = SmplxLiteJ24(gender="neutral")
            if device != "cpu":
                self.body_model = self.body_model.to(device)
            self.body_model.eval()

        # Instantiate checkers
        self.checkers: Dict[str, BaseQualityChecker] = {}
        for name in self.enabled:
            info = CHECKER_REGISTRY[name]
            kwargs = dict(self.checker_kwargs.get(name, {}))
            kwargs["device"] = device
            if info["needs_fk"]:
                kwargs["body_model"] = self.body_model
            self.checkers[name] = info["cls"](**kwargs)

    def check(self, motion: Union[Dict, str, Path]) -> AggregatedCheckResult:
        """Run all enabled checkers on a motion.

        Args:
            motion: Motion dict (with "poses", "trans") or path to .npz file.

        Returns:
            AggregatedCheckResult with combined verdict and per-checker details.
        """
        total_start = time.perf_counter()
        load_ms = 0.0
        # Load once, pass dict to all checkers
        if isinstance(motion, (str, Path)):
            path = Path(motion)
            if not path.exists():
                return AggregatedCheckResult(
                    is_valid=False,
                    failed_checks=["file_not_found"],
                    borderline_checks=[],
                    all_results={},
                    combined_reason=f"File not found: {path}",
                    profiling={"load_ms": 0.0, "precompute_ms": 0.0, "shared_fk_ms": 0.0, "checker_sum_ms": 0.0, "total_ms": 0.0},
                )
            load_start = time.perf_counter()
            data = dict(np.load(path, allow_pickle=True))
            load_ms = float((time.perf_counter() - load_start) * 1000.0)
            if "transl" in data and "trans" not in data:
                data["trans"] = data["transl"]
        else:
            data = dict(motion)
            if "transl" in data and "trans" not in data:
                data["trans"] = data["transl"]

        precompute_start = time.perf_counter()
        poses_3d = None
        if "poses" in data:
            try:
                poses_3d = normalize_poses_array(np.asarray(data["poses"]), num_joints=22)
                data["_cached_poses_22"] = poses_3d
            except Exception:
                poses_3d = None
        if poses_3d is not None:
            try:
                data["_cached_root_rot_mats_22"] = root_rotation_matrices_from_poses(poses_3d, device=self.device)
            except Exception:
                pass
        precompute_ms = float((time.perf_counter() - precompute_start) * 1000.0)

        shared_fk_ms = 0.0
        if self.body_model is not None and "poses" in data and "trans" in data:
            try:
                fk_start = time.perf_counter()
                data["_cached_joints_22"] = _compute_shared_fk_joints_22(
                    data,
                    self.body_model,
                    self.device,
                    poses_3d=poses_3d,
                )
                shared_fk_ms = float((time.perf_counter() - fk_start) * 1000.0)
            except Exception:
                # FK-less checkers can still run; FK-based ones will report their own failure.
                pass

        all_results: Dict[str, CheckResult] = {}
        failed_checks: List[str] = []
        borderline_checks: List[str] = []
        checker_sum_ms = 0.0
        for name, checker in self.checkers.items():
            start_time = time.perf_counter()
            try:
                result = checker.check(data)
            except Exception as e:
                result = CheckResult(
                    is_valid=False,
                    invalid_reason=f"Checker error: {e}",
                    invalid_mask=None,
                    details={"error": str(e)},
                    severity="fail",
                )
            elapsed_ms = float((time.perf_counter() - start_time) * 1000.0)
            checker_sum_ms += elapsed_ms
            details = dict(result.get("details") or {})
            details["elapsed_ms"] = elapsed_ms
            result["details"] = details
            result["severity"] = str(
                result.get("severity") or ("fail" if not result.get("is_valid", True) else "pass")
            ).strip().lower()
            if result["severity"] not in {"pass", "borderline", "fail"}:
                result["severity"] = "fail" if not result.get("is_valid", True) else "pass"
            try:
                result["invalid_mask"] = build_invalid_mask(name, checker, data, result)
            except Exception:
                # Keep checker failure reporting robust even if mask reconstruction fails.
                pass
            all_results[name] = result
            reason = result.get("invalid_reason", "")
            if "Too short" in reason or "Missing" in reason:
                continue
            if not result.get("is_valid", True):
                failed_checks.append(name)
            elif result.get("severity") == "borderline":
                borderline_checks.append(name)

        is_valid = len(failed_checks) == 0

        if failed_checks:
            reasons = []
            for name in failed_checks:
                r = all_results[name]
                reasons.append(f"[{name}] {r.get('invalid_reason', 'failed')}")
            if borderline_checks:
                mild = ", ".join(borderline_checks)
                reasons.append(f"borderline={mild}")
            combined_reason = "; ".join(reasons)
        elif borderline_checks:
            reasons = []
            for name in borderline_checks:
                r = all_results[name]
                reasons.append(f"[{name}] {r.get('invalid_reason', 'borderline')}")
            combined_reason = "Borderline issues: " + "; ".join(reasons)
        else:
            combined_reason = "All checks passed"

        total_ms = float((time.perf_counter() - total_start) * 1000.0)
        return AggregatedCheckResult(
            is_valid=is_valid,
            failed_checks=failed_checks,
            borderline_checks=borderline_checks,
            all_results=all_results,
            combined_reason=combined_reason,
            profiling={
                "load_ms": load_ms,
                "precompute_ms": precompute_ms,
                "shared_fk_ms": shared_fk_ms,
                "checker_sum_ms": checker_sum_ms,
                "total_ms": total_ms,
                "overhead_ms": max(total_ms - load_ms - precompute_ms - shared_fk_ms - checker_sum_ms, 0.0),
            },
        )

    def check_from_file(self, path: Union[str, Path]) -> AggregatedCheckResult:
        """Load motion from file and run all checks."""
        return self.check(path)

    def get_enabled_checkers(self) -> List[str]:
        """Return list of enabled checker names."""
        return list(self.checkers.keys())
