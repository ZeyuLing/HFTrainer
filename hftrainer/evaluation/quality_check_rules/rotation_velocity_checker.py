"""
Rotation velocity checker: detects per-joint rotation velocity spikes.

Ports and adapts the RotationVelocityChecker from scripts/m2m/filter_data/motion_checker.py
to work with axis-angle poses directly (no rotation matrix / local_rotations required).

Rule:
  - For each joint, compute per-frame rotation angle change (geodesic distance
    between consecutive frames' axis-angle rotations).
  - If the max rotation velocity for any joint exceeds its per-joint threshold,
    flag as invalid.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

from .base_checker import BaseQualityChecker, CheckResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_BODY_JOINTS = 22

JOINT_NAMES = [
    "MidHip",    # 0
    "LUpLeg",    # 1
    "RUpLeg",    # 2
    "spine",     # 3
    "LLeg",      # 4
    "RLeg",      # 5
    "spine1",    # 6
    "LFoot",     # 7
    "RFoot",     # 8
    "spine2",    # 9
    "LToeBase",  # 10
    "RToeBase",  # 11
    "Neck",      # 12
    "LShoulder", # 13
    "RShoulder", # 14
    "Head",      # 15
    "LArm",      # 16
    "RArm",      # 17
    "LForeArm",  # 18
    "RForeArm",  # 19
    "LHand",     # 20
    "RHand",     # 21
]

# Per-joint rotation velocity thresholds (degrees per frame).
# Relaxed from original to reduce false positives on fast/expressive motions.
DEFAULT_ROTATION_VEL_THRESHOLDS = {
    0: 120,   # MidHip
    1: 90,    # LUpLeg
    2: 90,    # RUpLeg
    3: 60,    # spine
    4: 60,    # LLeg
    5: 60,    # RLeg
    6: 60,    # spine1
    7: 60,    # LFoot
    8: 60,    # RFoot
    9: 60,    # spine2
    10: 90,   # LToeBase  (toes are naturally fast-moving)
    11: 90,   # RToeBase
    12: 60,   # Neck
    13: 60,   # LShoulder (collar bones can shift fast in arm motions)
    14: 60,   # RShoulder
    15: 60,   # Head
    16: 120,  # LArm
    17: 120,  # RArm
    18: 90,   # LForeArm
    19: 90,   # RForeArm
    20: 90,   # LHand
    21: 90,   # RHand
}

# Require at least this many frames exceeding threshold to flag a joint
DEFAULT_MIN_SPIKE_FRAMES = 2

# Minimum frames to require for checking
MIN_FRAMES_REQUIRED = 3

def _compute_rotation_velocities(poses_3d: np.ndarray) -> np.ndarray:
    """Compute per-joint rotation velocity in degrees per frame.

    Args:
        poses_3d: (F, J, 3) axis-angle.

    Returns:
        rot_vel: (F-1, J) rotation velocity in degrees.
    """
    F, J, _ = poses_3d.shape
    rot_mats = R.from_rotvec(poses_3d.reshape(-1, 3)).as_matrix().reshape(F, J, 3, 3)
    rel = np.matmul(np.swapaxes(rot_mats[:-1], -1, -2), rot_mats[1:])
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


class RotationVelocityChecker(BaseQualityChecker):
    """Detects per-joint rotation velocity spikes that exceed per-joint thresholds.
    Does not require a body model (works on axis-angle poses directly)."""

    name = "rotation_velocity"

    def __init__(
        self,
        body_model=None,
        device: str = "cpu",
        thresholds: Optional[Dict[int, float]] = None,
        min_spike_frames: int = DEFAULT_MIN_SPIKE_FRAMES,
    ) -> None:
        super().__init__(body_model=body_model, device=device)
        self.thresholds = thresholds or dict(DEFAULT_ROTATION_VEL_THRESHOLDS)
        self.min_spike_frames = min_spike_frames

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
                details={"has_spike": False, "reason": err},
            )

        poses = np.array(data["poses"])
        if len(poses) < MIN_FRAMES_REQUIRED:
            return CheckResult(
                is_valid=True,
                invalid_reason="Too short",
                invalid_mask=None,
                details={"has_spike": False, "reason": "Too short"},
            )

        try:
            poses_3d = self.normalize_poses(poses, NUM_BODY_JOINTS)
        except ValueError as e:
            return CheckResult(
                is_valid=False,
                invalid_reason=str(e),
                invalid_mask=None,
                details={"has_spike": False, "reason": str(e)},
            )

        rot_vel = _compute_rotation_velocities(poses_3d)  # (F-1, J)
        max_vel_per_joint = np.max(rot_vel, axis=0)  # (J,)

        violated_joints: List[Dict] = []
        for j in range(NUM_BODY_JOINTS):
            threshold = self.thresholds.get(j, 120.0)
            spike_frames = np.where(rot_vel[:, j] > threshold)[0].tolist()
            if len(spike_frames) >= self.min_spike_frames:
                violated_joints.append(
                    {
                        "joint_id": j,
                        "joint_name": JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"Joint_{j}",
                        "max_velocity_deg": float(max_vel_per_joint[j]),
                        "threshold_deg": float(threshold),
                        "spike_frames": spike_frames,
                    }
                )

        if not violated_joints:
            return CheckResult(
                is_valid=True,
                invalid_reason="No rotation velocity spike detected",
                invalid_mask=None,
                details={
                    "has_spike": False,
                    "max_vel_per_joint": {
                        JOINT_NAMES[j]: float(max_vel_per_joint[j])
                        for j in range(min(NUM_BODY_JOINTS, len(JOINT_NAMES)))
                    },
                    "reason": "No rotation velocity spike detected",
                },
            )

        joint_info = [
            f"{d['joint_name']}({d['max_velocity_deg']:.1f}>{d['threshold_deg']:.0f})"
            for d in violated_joints
        ]
        reason = f"Rotation velocity spike: {', '.join(joint_info)}"
        details = {
            "has_spike": True,
            "violated_joints": violated_joints,
            "reason": reason,
        }
        return CheckResult(
            is_valid=False,
            invalid_reason=reason,
            invalid_mask=None,
            details=details,
        )


def detect_rotation_velocity(data: Dict, **kwargs) -> Dict:
    """Legacy API. Returns dict with keys: has_spike, violated_joints, reason."""
    checker = RotationVelocityChecker(**kwargs)
    result = checker.check(data)
    details = result.get("details") or {}
    return {
        "has_spike": details.get("has_spike", False),
        "violated_joints": details.get("violated_joints", []),
        "reason": details.get("reason", result.get("invalid_reason", "")),
    }
