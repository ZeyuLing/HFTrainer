"""
Base class for motion quality checkers.

All concrete checkers (e.g. JitterChecker, JointTwistChecker) inherit from
BaseQualityChecker and implement check(). Checkers can be used both for
filtering low-quality data and for evaluating repair results.
"""

from pathlib import Path
from typing import Any, Dict, Optional, TypedDict, Union

import numpy as np

# Optional: only import when a checker needs FK
from ._model_compat import SmplxLiteJ24


class CheckResult(TypedDict, total=False):
    """Result of a single quality check.

    Attributes:
        is_valid: True if the motion passes the check, False if it fails (low quality).
        invalid_reason: Short identifier or human-readable reason when is_valid is False
            (e.g. "jitter", "joint_twist"). When is_valid is True, typically empty or "No ... detected".
        invalid_mask: Optional [T, J] boolean or float mask indicating which frame/joint
            failed. T = number of frames, J = number of body joints (e.g. 22). None if
            the checker does not provide per-frame/joint granularity.
        details: Optional dict with checker-specific details (e.g. jitter_windows,
            twisted_joints). Used for reporting and for compatibility with legacy scripts.
        severity: One of ``pass``, ``borderline``, ``fail``. ``is_valid`` remains
            the hard-fail signal for backward compatibility; ``severity`` adds a
            mild-issue tier without forcing the sample into low quality.
    """

    is_valid: bool
    invalid_reason: str
    invalid_mask: Optional[np.ndarray]
    details: Optional[Dict[str, Any]]
    severity: str


# Default number of body joints used by most checkers (SMPL-H body without fingers).
NUM_BODY_JOINTS_DEFAULT = 22


def normalize_poses_array(
    poses: np.ndarray,
    num_joints: int = NUM_BODY_JOINTS_DEFAULT,
) -> np.ndarray:
    """Convert poses to (F, J, 3) axis-angle format."""
    if poses.ndim == 2:
        F, D = poses.shape
        if D >= 72:
            poses_3d = poses[:, :72].reshape(F, 24, 3)[:, :num_joints, :]
        elif D >= 66:
            poses_3d = poses[:, :66].reshape(F, 22, 3)
            if num_joints < 22:
                poses_3d = poses_3d[:, :num_joints, :]
        else:
            n = D // 3
            if n < num_joints:
                raise ValueError(f"Poses dimension too small: D={D}, need at least {num_joints * 3}")
            poses_3d = poses[:, : num_joints * 3].reshape(F, num_joints, 3)
    elif poses.ndim == 3:
        _, J, _ = poses.shape
        if J < num_joints:
            raise ValueError(f"Not enough joints: J={J}, need at least {num_joints}")
        poses_3d = poses[:, :num_joints, :].copy()
    else:
        raise ValueError(f"Unsupported poses shape: {poses.shape}")
    return poses_3d


def normalize_betas_array(betas: Any, max_dim: int = 16) -> Optional[np.ndarray]:
    if betas is None:
        return None
    arr = np.asarray(betas)
    if arr.ndim == 1:
        return arr[:max_dim]
    return arr.reshape(-1, max_dim)[:1]


class BaseQualityChecker:
    """Base class for all motion quality checkers.

    Subclasses must implement check(). Optionally override load_motion(),
    normalize_poses(), or use the optional body model for FK-based checks.
    """

    # Short identifier for this checker (e.g. "jitter", "joint_twist"). Override in subclass.
    name: str = "base"

    def __init__(
        self,
        body_model: Optional[Any] = None,
        device: str = "cuda",
    ) -> None:
        """Initialize the checker.

        Args:
            body_model: Optional SMPL/SMPL-X body model for forward kinematics.
                Pass None for checkers that do not need FK (e.g. joint twist).
            device: Device string for model and tensor ops ("cuda", "cuda:0", "cpu").
        """
        self.device = device
        self.body_model: Optional[Any] = None
        self.init_body_model(body_model)

    def init_body_model(self, body_model: Optional[Any] = None) -> None:
        """Set or clear the body model used for FK.

        Args:
            body_model: Optional body model instance. If None, self.body_model is
                left unchanged (subclasses may set a default).
        """
        if body_model is not None:
            self.body_model = body_model if self.device == "cpu" else body_model.to(self.device)

    def load_motion(self, motion: Union[str, Path, Dict]) -> Dict:
        """Load motion data from a path or return a dict as-is.

        Args:
            motion: Either a path to an .npz file (str or Path) or an in-memory
                dict with keys such as "poses", "trans", "betas".

        Returns:
            Motion dict with at least "poses" and "trans" (or "transl").
            Keys are normalized (e.g. "trans" used for root translation).

        Raises:
            FileNotFoundError: If motion is a path and the file does not exist.
            ValueError: If the loaded object is not a dict or lacks required keys.
        """
        if isinstance(motion, (str, Path)):
            path = Path(motion)
            if not path.exists():
                raise FileNotFoundError(f"Motion file not found: {path}")
            data = dict(np.load(path, allow_pickle=True))
        elif isinstance(motion, dict):
            data = dict(motion)
        else:
            raise ValueError("motion must be a path (str or Path) or a dict")

        # Normalize key names if present
        if "transl" in data and "trans" not in data:
            data["trans"] = data["transl"]
        return data

    def normalize_poses(
        self,
        poses: np.ndarray,
        num_joints: int = NUM_BODY_JOINTS_DEFAULT,
    ) -> np.ndarray:
        """Convert poses to (F, J, 3) axis-angle format.

        Handles flattened (F, D) and already 3D (F, J, 3) inputs. Ensures
        exactly num_joints joints (truncation or padding not applied; caller
        must ensure enough dimensions).

        Args:
            poses: (F, D) with D = J*3, or (F, J, 3). May be 72 (24 joints) or 66 (22).

        Returns:
            (F, J, 3) array with J = num_joints.

        Raises:
            ValueError: If poses shape is unsupported or has fewer than num_joints joints.
        """
        return normalize_poses_array(poses, num_joints=num_joints)

    def check(self, motion: Union[Dict, str, Path]) -> CheckResult:
        """Run the quality check on a single motion.

        Must be implemented by subclasses.

        Args:
            motion: Motion dict (with "poses", "trans") or path to .npz.

        Returns:
            CheckResult with is_valid, invalid_reason, and optionally
            invalid_mask and details.
        """
        raise NotImplementedError("Subclasses must implement check().")

    def check_from_file(self, path: Union[str, Path]) -> CheckResult:
        """Load motion from file and run check. Convenience wrapper.

        Args:
            path: Path to .npz motion file.

        Returns:
            CheckResult from check(load_motion(path)).
        """
        data = self.load_motion(path)
        return self.check(data)

    def get_required_keys(self) -> list:
        """Return list of keys that must be present in the motion dict for check().

        Override in subclass if different from ["poses", "trans"].
        """
        return ["poses", "trans"]

    def validate_motion_dict(self, data: Dict) -> Optional[str]:
        """Validate that data has required keys and basic shape. Call at start of check().

        Returns:
            None if valid, else an error message string (invalid_reason).
        """
        for key in self.get_required_keys():
            if key not in data:
                return f"Missing {key}"
        poses = np.asarray(data["poses"])
        if poses.size == 0:
            return "Empty poses"
        return None
