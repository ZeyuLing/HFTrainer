"""ML-based rotation validity checker for 21 body joints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
import torch

from ._geometry_compat import axis_angle_to_matrix
from ._model_compat import JointRotationClassifier

from .base_checker import BaseQualityChecker, CheckResult
from .joint_twist_checker import JOINT_TWIST_AXIS_MAP, extract_joint_twist_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_CHECKED_JOINTS = list(range(1, 22))
_JOINT_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
]

_HEURISTIC_LIMITS = {
    1: {"angle_borderline": 175.0, "angle_fail": 195.0, "twist_borderline": 150.0, "twist_fail": 170.0},
    2: {"angle_borderline": 175.0, "angle_fail": 195.0, "twist_borderline": 150.0, "twist_fail": 170.0},
    3: {"angle_borderline": 105.0, "angle_fail": 125.0},
    4: {"angle_borderline": 185.0, "angle_fail": 205.0, "twist_borderline": 145.0, "twist_fail": 165.0},
    5: {"angle_borderline": 185.0, "angle_fail": 205.0, "twist_borderline": 145.0, "twist_fail": 165.0},
    6: {"angle_borderline": 105.0, "angle_fail": 125.0},
    7: {"angle_borderline": 165.0, "angle_fail": 185.0, "twist_borderline": 125.0, "twist_fail": 145.0},
    8: {"angle_borderline": 165.0, "angle_fail": 185.0, "twist_borderline": 125.0, "twist_fail": 145.0},
    9: {"angle_borderline": 95.0, "angle_fail": 115.0},
    10: {"angle_borderline": 145.0, "angle_fail": 165.0},
    11: {"angle_borderline": 145.0, "angle_fail": 165.0},
    12: {"angle_borderline": 115.0, "angle_fail": 135.0},
    13: {"angle_borderline": 130.0, "angle_fail": 150.0, "twist_borderline": 120.0, "twist_fail": 140.0},
    14: {"angle_borderline": 130.0, "angle_fail": 150.0, "twist_borderline": 120.0, "twist_fail": 140.0},
    15: {"angle_borderline": 110.0, "angle_fail": 130.0},
    16: {"angle_borderline": 185.0, "angle_fail": 205.0, "twist_borderline": 150.0, "twist_fail": 170.0},
    17: {"angle_borderline": 185.0, "angle_fail": 205.0, "twist_borderline": 150.0, "twist_fail": 170.0},
    18: {"angle_borderline": 195.0, "angle_fail": 215.0, "twist_borderline": 155.0, "twist_fail": 175.0},
    19: {"angle_borderline": 195.0, "angle_fail": 215.0, "twist_borderline": 155.0, "twist_fail": 175.0},
    20: {"angle_borderline": 205.0, "angle_fail": 225.0, "twist_borderline": 170.0, "twist_fail": 190.0},
    21: {"angle_borderline": 205.0, "angle_fail": 225.0, "twist_borderline": 170.0, "twist_fail": 190.0},
}

_PERSISTENCE_FAIL_RUN = 8
_PERSISTENCE_BORDERLINE_RUN = 5
_FAIL_RATIO_THRESHOLD = 0.12
_BORDERLINE_RATIO_THRESHOLD = 0.04


def _default_thresholds() -> Dict[str, float]:
    return {f"joint_{joint_id}": 0.5 for joint_id in _CHECKED_JOINTS}


def _longest_true_run(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size == 0 or not mask.any():
        return 0
    padded = np.pad(mask.astype(np.int32), (1, 1), constant_values=0)
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    if starts.size == 0 or ends.size == 0:
        return 0
    return int(np.max(ends - starts))


def _keep_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size == 0 or not mask.any():
        return np.zeros_like(mask, dtype=bool)
    padded = np.pad(mask.astype(np.int32), (1, 1), constant_values=0)
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    kept = np.zeros_like(mask, dtype=bool)
    for start, end in zip(starts, ends):
        if int(end - start) >= int(min_run):
            kept[start:end] = True
    return kept


class RotationValidityChecker(BaseQualityChecker):
    """Checks whether local joint rotations look anatomically valid."""

    name = "rotation_validity"

    _classifier: ClassVar[Optional[JointRotationClassifier]] = None
    _classifier_path: ClassVar[Optional[str]] = None
    _resolved_auto_path: ClassVar[Optional[str]] = None
    _auto_search_done: ClassVar[bool] = False

    def __init__(
        self,
        model_path: str = "auto",
        thresholds: Optional[Dict[str, float]] = None,
        device: str = "cuda",
        fallback_mode: str = "skip",
        **kwargs: Any,
    ) -> None:
        super().__init__(body_model=None, device=device)
        self.model_path = model_path
        self.fallback_mode = str(fallback_mode or "skip").strip().lower()
        self.thresholds = dict(_default_thresholds())
        if thresholds:
            self.thresholds.update({str(k): float(v) for k, v in thresholds.items()})

    def get_required_keys(self) -> list:
        return ["poses"]

    def check(self, motion) -> CheckResult:
        data = self.load_motion(motion)
        err = self.validate_motion_dict(data)
        if err:
            return CheckResult(
                is_valid=False,
                invalid_reason=err,
                invalid_mask=None,
                details={"error": err},
            )

        poses = np.asarray(data["poses"], dtype=np.float32)
        num_frames = int(poses.shape[0]) if poses.ndim >= 1 else 0

        classifier = self._get_classifier(self.model_path, self.device)
        if classifier is None:
            if self.fallback_mode == "skip":
                return CheckResult(
                    is_valid=True,
                    invalid_reason="Rotation validity classifier unavailable, skipped",
                    invalid_mask=np.zeros((max(num_frames, 0), 22), dtype=bool),
                    details={
                        "skipped": True,
                        "heuristic": False,
                        "classifier_unavailable": True,
                        "model_path": self.model_path,
                        "fallback_mode": self.fallback_mode,
                    },
                    severity="pass",
                )
            return self._heuristic_check(poses=poses, model_path=self.model_path)

        try:
            poses_3d = self.normalize_poses(poses, num_joints=22)
        except Exception as exc:
            return CheckResult(
                is_valid=False,
                invalid_reason=f"Invalid poses: {exc}",
                invalid_mask=None,
                details={"error": str(exc)},
            )

        num_frames = int(poses_3d.shape[0])
        if num_frames <= 0:
            return CheckResult(
                is_valid=True,
                invalid_reason="Empty motion, skipped",
                invalid_mask=np.zeros((0, 22), dtype=bool),
                details={"skipped": True},
            )

        rot_mats = axis_angle_to_matrix(torch.as_tensor(poses_3d, dtype=torch.float32)).cpu().numpy()
        available_joints = set(classifier.get_available_joints())
        checked_joints = [joint_id for joint_id in _CHECKED_JOINTS if joint_id in available_joints]
        if not checked_joints:
            return CheckResult(
                is_valid=True,
                invalid_reason="Rotation validity classifier has no matching joints, skipped",
                invalid_mask=np.zeros((num_frames, 22), dtype=bool),
                details={"skipped": True, "available_joints": sorted(available_joints)},
            )

        probs_matrix, _ = classifier.predict_all_joints_batch(checked_joints, rot_mats)
        invalid_mask = np.zeros((num_frames, 22), dtype=bool)
        failed_checks: List[str] = []
        borderline_checks: List[str] = []
        joint_metrics: Dict[str, Dict[str, float]] = {}
        failed_joint_ids: List[int] = []
        borderline_joint_ids: List[int] = []

        for idx, joint_id in enumerate(checked_joints):
            probs = probs_matrix[:, idx]
            threshold = float(self.thresholds.get(f"joint_{joint_id}", 0.5))
            mean_prob = float(probs.mean()) if probs.size else 0.0
            min_prob = float(probs.min()) if probs.size else 0.0
            invalid_frames = probs < threshold
            invalid_ratio = float(invalid_frames.mean()) if probs.size else 0.0
            longest_run = _longest_true_run(invalid_frames)
            severe_frames = probs < max(threshold * 0.7, threshold - 0.2)
            severe_ratio = float(severe_frames.mean()) if probs.size else 0.0
            joint_metrics[str(joint_id)] = {
                "mean_valid_prob": mean_prob,
                "min_valid_prob": min_prob,
                "threshold": threshold,
                "invalid_frame_count": int(invalid_frames.sum()),
                "invalid_ratio": invalid_ratio,
                "longest_invalid_run": longest_run,
                "severe_ratio": severe_ratio,
            }
            if not invalid_frames.any():
                continue
            joint_name = _JOINT_NAMES[joint_id] if joint_id < len(_JOINT_NAMES) else f"Joint_{joint_id}"
            if (
                severe_ratio >= 0.02
                or longest_run >= _PERSISTENCE_FAIL_RUN
                or invalid_ratio >= _FAIL_RATIO_THRESHOLD
                or mean_prob < max(0.18, threshold - 0.18)
            ):
                invalid_mask[:, joint_id] = invalid_frames
                failed_joint_ids.append(joint_id)
                failed_checks.append(f"{joint_name}({joint_id}) validity fail")
            elif (
                longest_run >= _PERSISTENCE_BORDERLINE_RUN
                or invalid_ratio >= _BORDERLINE_RATIO_THRESHOLD
                or mean_prob < threshold
            ):
                invalid_mask[:, joint_id] = invalid_frames
                borderline_joint_ids.append(joint_id)
                borderline_checks.append(f"{joint_name}({joint_id}) validity borderline")

        return CheckResult(
            is_valid=len(failed_joint_ids) == 0,
            invalid_reason=(
                "; ".join(failed_checks)
                if failed_checks
                else ("; ".join(borderline_checks) if borderline_checks else "All joints passed rotation validity")
            ),
            invalid_mask=invalid_mask,
            details={
                "checked_joints": checked_joints,
                "failed_joint_ids": failed_joint_ids,
                "borderline_joint_ids": borderline_joint_ids,
                "joint_metrics": joint_metrics,
                "model_path": self._classifier_path or self.model_path,
                "heuristic": False,
            },
            severity="fail" if failed_joint_ids else ("borderline" if borderline_joint_ids else "pass"),
        )

    def _heuristic_check(self, poses: np.ndarray, model_path: str) -> CheckResult:
        try:
            poses_3d = self.normalize_poses(poses, num_joints=22)
        except Exception as exc:
            return CheckResult(
                is_valid=False,
                invalid_reason=f"Invalid poses: {exc}",
                invalid_mask=None,
                details={"error": str(exc)},
            )
        num_frames = int(poses_3d.shape[0])
        if num_frames <= 0:
            return CheckResult(
                is_valid=True,
                invalid_reason="Empty motion, skipped",
                invalid_mask=np.zeros((0, 22), dtype=bool),
                details={"skipped": True, "heuristic": True},
            )

        aa_deg = np.rad2deg(poses_3d)
        angle_mag = np.linalg.norm(aa_deg, axis=-1)
        invalid_mask = np.zeros((num_frames, 22), dtype=bool)
        failed_joint_ids: List[int] = []
        borderline_joint_ids: List[int] = []
        failed_checks: List[str] = []
        borderline_checks: List[str] = []
        joint_metrics: Dict[str, Dict[str, float]] = {}
        for joint_id, limits in _HEURISTIC_LIMITS.items():
            if joint_id >= poses_3d.shape[1]:
                continue
            total_angle_deg = np.asarray(angle_mag[:, joint_id], dtype=np.float32)
            twist_deg = np.zeros((num_frames,), dtype=np.float32)
            if joint_id in JOINT_TWIST_AXIS_MAP:
                twist_metrics = extract_joint_twist_metrics(poses_3d[:, joint_id, :], JOINT_TWIST_AXIS_MAP[joint_id])
                twist_deg = np.abs(np.rad2deg(twist_metrics["geometric_twist_rad"])).astype(np.float32)

            borderline_frames_raw = total_angle_deg > float(limits["angle_borderline"])
            fail_frames_raw = total_angle_deg > float(limits["angle_fail"])
            if "twist_borderline" in limits:
                borderline_frames_raw |= twist_deg > float(limits["twist_borderline"])
            if "twist_fail" in limits:
                fail_frames_raw |= twist_deg > float(limits["twist_fail"])

            fail_frames = _keep_runs(fail_frames_raw, _PERSISTENCE_FAIL_RUN)
            borderline_frames = _keep_runs(borderline_frames_raw, _PERSISTENCE_BORDERLINE_RUN) & (~fail_frames)
            invalid_count = int((fail_frames | borderline_frames).sum())
            invalid_ratio = float(invalid_count / max(num_frames, 1))
            longest_fail_run = _longest_true_run(fail_frames)
            longest_borderline_run = _longest_true_run(borderline_frames)
            joint_metrics[str(joint_id)] = {
                "invalid_frame_count": invalid_count,
                "invalid_ratio": invalid_ratio,
                "longest_fail_run": longest_fail_run,
                "longest_borderline_run": longest_borderline_run,
                "q95_angle_deg": float(np.percentile(total_angle_deg, 95.0)) if num_frames > 0 else 0.0,
                "q99_angle_deg": float(np.percentile(total_angle_deg, 99.0)) if num_frames > 0 else 0.0,
                "max_angle_deg": float(total_angle_deg.max()) if num_frames > 0 else 0.0,
                "q95_twist_deg": float(np.percentile(twist_deg, 95.0)) if num_frames > 0 else 0.0,
                "q99_twist_deg": float(np.percentile(twist_deg, 99.0)) if num_frames > 0 else 0.0,
                "max_twist_deg": float(twist_deg.max()) if num_frames > 0 else 0.0,
                "limit_angle_borderline_deg": float(limits["angle_borderline"]),
                "limit_angle_fail_deg": float(limits["angle_fail"]),
                "limit_twist_borderline_deg": float(limits.get("twist_borderline", 0.0)),
                "limit_twist_fail_deg": float(limits.get("twist_fail", 0.0)),
            }
            if invalid_count <= 0:
                continue
            joint_name = _JOINT_NAMES[joint_id] if joint_id < len(_JOINT_NAMES) else f"Joint_{joint_id}"
            if (
                fail_frames.any()
                and (
                    longest_fail_run >= _PERSISTENCE_FAIL_RUN
                    or float(fail_frames.mean()) >= _FAIL_RATIO_THRESHOLD
                )
            ):
                invalid_mask[:, joint_id] = fail_frames
                failed_joint_ids.append(joint_id)
                failed_checks.append(f"{joint_name}({joint_id}) extreme heuristic fail")
            elif (
                borderline_frames.any()
                and (
                    longest_borderline_run >= _PERSISTENCE_BORDERLINE_RUN
                    or float(borderline_frames.mean()) >= _BORDERLINE_RATIO_THRESHOLD
                )
            ):
                invalid_mask[:, joint_id] = borderline_frames
                borderline_joint_ids.append(joint_id)
                borderline_checks.append(f"{joint_name}({joint_id}) extreme heuristic borderline")

        return CheckResult(
            is_valid=len(failed_joint_ids) == 0,
            invalid_reason=(
                "; ".join(failed_checks)
                if failed_checks
                else (
                    "; ".join(borderline_checks)
                    if borderline_checks
                    else "All joints passed heuristic rotation validity"
                )
            ),
            invalid_mask=invalid_mask,
            details={
                "checked_joints": sorted(_HEURISTIC_LIMITS.keys()),
                "failed_joint_ids": failed_joint_ids,
                "borderline_joint_ids": borderline_joint_ids,
                "joint_metrics": joint_metrics,
                "model_path": model_path,
                "heuristic": True,
                "classifier_unavailable": True,
                "fallback_mode": self.fallback_mode,
            },
            severity="fail" if failed_joint_ids else ("borderline" if borderline_joint_ids else "pass"),
        )

    @classmethod
    def _get_classifier(cls, model_path: str, device: str) -> Optional[JointRotationClassifier]:
        resolved = cls._resolve_model_path(model_path)
        if not resolved:
            return None
        if cls._classifier is not None and cls._classifier_path == resolved:
            return cls._classifier
        try:
            classifier = JointRotationClassifier(
                resolved,
                device=torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu"),
            )
        except Exception:
            return None
        cls._classifier = classifier
        cls._classifier_path = resolved
        return classifier

    @classmethod
    def _resolve_model_path(cls, model_path: str) -> Optional[str]:
        if model_path and model_path != "auto":
            explicit = Path(model_path)
            if not explicit.is_absolute():
                explicit = PROJECT_ROOT / explicit
            return str(explicit.resolve()) if explicit.is_file() else None

        if cls._resolved_auto_path is not None:
            return cls._resolved_auto_path
        if cls._auto_search_done:
            return None

        cls._auto_search_done = True
        candidates: List[Path] = []

        env_path = os.environ.get("HYMOTION_ROTATION_VALIDITY_MODEL", "").strip()
        if env_path:
            env_candidate = Path(env_path)
            if not env_candidate.is_absolute():
                env_candidate = PROJECT_ROOT / env_candidate
            if env_candidate.is_file():
                cls._resolved_auto_path = str(env_candidate.resolve())
                return cls._resolved_auto_path

        search_dirs = [
            PROJECT_ROOT / "data/annotations/trained_classifiers",
            PROJECT_ROOT / "data/trained_classifiers",
            PROJECT_ROOT / "scripts/joint_rotation_checker",
            PROJECT_ROOT / "scripts/joint_rotation_checker/checkpoints",
            PROJECT_ROOT / "scripts/joint_rotation_checker/models",
            PROJECT_ROOT / "output/joint_rotation_checker",
            PROJECT_ROOT / "output/rotation_validity",
        ]
        for directory in search_dirs:
            if not directory.is_dir():
                continue
            candidates.extend(directory.glob("joint_classifiers_*.pt"))
            candidates.extend(directory.glob("**/joint_classifiers_*.pt"))

        if not candidates:
            return None
        candidates = sorted({path.resolve() for path in candidates})
        cls._resolved_auto_path = str(candidates[-1])
        return cls._resolved_auto_path
