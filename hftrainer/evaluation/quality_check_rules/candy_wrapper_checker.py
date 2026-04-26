from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .base_checker import BaseQualityChecker, CheckResult
from .tbs_utils import JOINT_NAMES, extract_joint_tbs_metrics

NUM_BODY_JOINTS = 22
MIN_FRAMES_REQUIRED = 6

ARM_SIDES = [
    {"side": "left", "collar": 13, "shoulder": 16, "elbow": 18, "wrist": 20},
    {"side": "right", "collar": 14, "shoulder": 17, "elbow": 19, "wrist": 21},
]


def _sustained_frames(frame_mask: np.ndarray, min_frames: int, min_ratio: float) -> List[int]:
    indices = np.where(np.asarray(frame_mask, dtype=bool))[0].astype(int).tolist()
    if len(indices) < int(min_frames):
        return []
    if len(indices) < float(frame_mask.shape[0]) * float(min_ratio):
        return []
    return indices


def _safe_mean(values: np.ndarray, frame_indices: List[int]) -> float:
    if not frame_indices:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr[np.asarray(frame_indices, dtype=np.int64)]))


class CandyWrapperChecker(BaseQualityChecker):
    """Detect common wrist/elbow/shoulder candy-wrapper patterns in TBS space."""

    name = "candy_wrapper"

    def __init__(
        self,
        wrist_flip_abs_deg: float = 140.0,
        wrist_flip_min_frames: int = 4,
        wrist_flip_min_ratio: float = 0.03,
        elbow_counter_min_abs_deg: float = 45.0,
        wrist_counter_min_abs_deg: float = 100.0,
        counter_balance_margin_deg: float = 55.0,
        counter_min_frames: int = 5,
        counter_min_ratio: float = 0.04,
        chain_parent_min_abs_deg: float = 90.0,
        chain_wrist_min_abs_deg: float = 100.0,
        chain_balance_margin_deg: float = 60.0,
        chain_min_frames: int = 5,
        chain_min_ratio: float = 0.04,
        device: str = "cuda",
    ) -> None:
        super().__init__(body_model=None, device=device)
        self.wrist_flip_abs_deg = float(wrist_flip_abs_deg)
        self.wrist_flip_min_frames = int(wrist_flip_min_frames)
        self.wrist_flip_min_ratio = float(wrist_flip_min_ratio)
        self.elbow_counter_min_abs_deg = float(elbow_counter_min_abs_deg)
        self.wrist_counter_min_abs_deg = float(wrist_counter_min_abs_deg)
        self.counter_balance_margin_deg = float(counter_balance_margin_deg)
        self.counter_min_frames = int(counter_min_frames)
        self.counter_min_ratio = float(counter_min_ratio)
        self.chain_parent_min_abs_deg = float(chain_parent_min_abs_deg)
        self.chain_wrist_min_abs_deg = float(chain_wrist_min_abs_deg)
        self.chain_balance_margin_deg = float(chain_balance_margin_deg)
        self.chain_min_frames = int(chain_min_frames)
        self.chain_min_ratio = float(chain_min_ratio)

    def get_required_keys(self) -> list:
        return ["poses"]

    def check(self, motion: Union[Dict, str, Path]) -> CheckResult:
        if isinstance(motion, (str, Path)):
            data = self.load_motion(motion)
        else:
            data = dict(motion)

        err = self.validate_motion_dict(data)
        if err is not None:
            return CheckResult(
                is_valid=False,
                invalid_reason=err,
                invalid_mask=None,
                details={"has_candy_wrapper": False, "reason": err},
            )

        poses = np.asarray(data["poses"])
        try:
            poses_3d = np.asarray(data.get("_cached_poses_22")) if data.get("_cached_poses_22") is not None else None
            if poses_3d is None:
                poses_3d = self.normalize_poses(poses, NUM_BODY_JOINTS)
        except ValueError as exc:
            return CheckResult(
                is_valid=False,
                invalid_reason=str(exc),
                invalid_mask=None,
                details={"has_candy_wrapper": False, "reason": str(exc)},
            )

        num_frames = int(poses_3d.shape[0])
        if num_frames < MIN_FRAMES_REQUIRED:
            reason = f"Too short (need at least {MIN_FRAMES_REQUIRED} frames)"
            return CheckResult(
                is_valid=True,
                invalid_reason=reason,
                invalid_mask=np.zeros((num_frames, NUM_BODY_JOINTS), dtype=bool),
                details={"has_candy_wrapper": False, "reason": reason, "events": []},
                severity="pass",
            )

        invalid_mask = np.zeros((num_frames, NUM_BODY_JOINTS), dtype=bool)
        events: List[Dict[str, object]] = []
        failed_joint_ids: List[int] = []

        for side_cfg in ARM_SIDES:
            collar_id = int(side_cfg["collar"])
            shoulder_id = int(side_cfg["shoulder"])
            elbow_id = int(side_cfg["elbow"])
            wrist_id = int(side_cfg["wrist"])

            collar_twist = extract_joint_tbs_metrics(poses_3d[:, collar_id, :], collar_id)["twist_deg"]
            shoulder_twist = extract_joint_tbs_metrics(poses_3d[:, shoulder_id, :], shoulder_id)["twist_deg"]
            elbow_twist = extract_joint_tbs_metrics(poses_3d[:, elbow_id, :], elbow_id)["twist_deg"]
            wrist_twist = extract_joint_tbs_metrics(poses_3d[:, wrist_id, :], wrist_id)["twist_deg"]

            wrist_flip_frames = _sustained_frames(
                np.abs(wrist_twist) >= self.wrist_flip_abs_deg,
                self.wrist_flip_min_frames,
                self.wrist_flip_min_ratio,
            )
            if wrist_flip_frames:
                invalid_mask[np.asarray(wrist_flip_frames, dtype=np.int64), wrist_id] = True
                failed_joint_ids.append(wrist_id)
                events.append(
                    {
                        "type": "wrist_flip_180",
                        "side": side_cfg["side"],
                        "frame_indices": wrist_flip_frames,
                        "joint_ids": [wrist_id],
                        "joint_names": [JOINT_NAMES[wrist_id]],
                        "mean_wrist_twist_deg": _safe_mean(np.abs(wrist_twist), wrist_flip_frames),
                    }
                )

            elbow_counter_mask = (
                (np.sign(elbow_twist) * np.sign(wrist_twist) < 0.0)
                & (np.abs(elbow_twist) >= self.elbow_counter_min_abs_deg)
                & (np.abs(wrist_twist) >= self.wrist_counter_min_abs_deg)
                & (np.abs(elbow_twist + wrist_twist) <= self.counter_balance_margin_deg)
            )
            elbow_counter_frames = _sustained_frames(
                elbow_counter_mask,
                self.counter_min_frames,
                self.counter_min_ratio,
            )
            if elbow_counter_frames:
                idx = np.asarray(elbow_counter_frames, dtype=np.int64)
                invalid_mask[idx, elbow_id] = True
                invalid_mask[idx, wrist_id] = True
                failed_joint_ids.extend([elbow_id, wrist_id])
                events.append(
                    {
                        "type": "elbow_wrist_counter_twist",
                        "side": side_cfg["side"],
                        "frame_indices": elbow_counter_frames,
                        "joint_ids": [elbow_id, wrist_id],
                        "joint_names": [JOINT_NAMES[elbow_id], JOINT_NAMES[wrist_id]],
                        "mean_elbow_twist_deg": _safe_mean(np.abs(elbow_twist), elbow_counter_frames),
                        "mean_wrist_twist_deg": _safe_mean(np.abs(wrist_twist), elbow_counter_frames),
                        "mean_balance_deg": _safe_mean(np.abs(elbow_twist + wrist_twist), elbow_counter_frames),
                    }
                )

            proximal_sum = collar_twist + shoulder_twist + elbow_twist
            chain_counter_mask = (
                (np.sign(proximal_sum) * np.sign(wrist_twist) < 0.0)
                & (np.abs(proximal_sum) >= self.chain_parent_min_abs_deg)
                & (np.abs(wrist_twist) >= self.chain_wrist_min_abs_deg)
                & (np.abs(proximal_sum + wrist_twist) <= self.chain_balance_margin_deg)
            )
            chain_frames = _sustained_frames(
                chain_counter_mask,
                self.chain_min_frames,
                self.chain_min_ratio,
            )
            if chain_frames:
                idx = np.asarray(chain_frames, dtype=np.int64)
                frame_joint_mask = np.stack(
                    [
                        np.abs(collar_twist[idx]) >= 20.0,
                        np.abs(shoulder_twist[idx]) >= 20.0,
                        np.abs(elbow_twist[idx]) >= 20.0,
                        np.abs(wrist_twist[idx]) >= 20.0,
                    ],
                    axis=1,
                )
                chain_joint_ids = [collar_id, shoulder_id, elbow_id, wrist_id]
                for local_row, frame_idx in enumerate(idx.tolist()):
                    active_joint_ids = [chain_joint_ids[j] for j, active in enumerate(frame_joint_mask[local_row]) if bool(active)]
                    if not active_joint_ids:
                        active_joint_ids = chain_joint_ids
                    invalid_mask[frame_idx, active_joint_ids] = True
                failed_joint_ids.extend(chain_joint_ids)
                events.append(
                    {
                        "type": "distributed_chain_compensation",
                        "side": side_cfg["side"],
                        "frame_indices": chain_frames,
                        "joint_ids": chain_joint_ids,
                        "joint_names": [JOINT_NAMES[j] for j in chain_joint_ids],
                        "mean_proximal_sum_deg": _safe_mean(np.abs(proximal_sum), chain_frames),
                        "mean_wrist_twist_deg": _safe_mean(np.abs(wrist_twist), chain_frames),
                        "mean_chain_balance_deg": _safe_mean(np.abs(proximal_sum + wrist_twist), chain_frames),
                    }
                )

        if not events:
            return CheckResult(
                is_valid=True,
                invalid_reason="No candy wrapper detected",
                invalid_mask=np.zeros((num_frames, NUM_BODY_JOINTS), dtype=bool),
                details={
                    "has_candy_wrapper": False,
                    "events": [],
                    "failed_joint_ids": [],
                    "coordinate_system": "tbs",
                    "reason": "No candy wrapper detected",
                },
                severity="pass",
            )

        events.sort(key=lambda item: (min(item.get("frame_indices") or [0]), str(item.get("type", ""))))
        failed_joint_ids = sorted(set(int(j) for j in failed_joint_ids))
        reason = "Candy wrapper detected: " + ", ".join(
            f"{item['side']}:{item['type']}" for item in events
        )
        return CheckResult(
            is_valid=False,
            invalid_reason=reason,
            invalid_mask=invalid_mask,
            details={
                "has_candy_wrapper": True,
                "events": events,
                "failed_joint_ids": failed_joint_ids,
                "coordinate_system": "tbs",
                "reason": reason,
            },
            severity="fail",
        )

