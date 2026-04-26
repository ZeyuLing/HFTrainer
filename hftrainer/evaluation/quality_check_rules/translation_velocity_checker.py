"""
Translation velocity checker: detects root translation velocity spikes and outliers.

Combines two checks ported from scripts/m2m/filter_data/motion_checker.py:
1. TransVelocityChecker: per-axis absolute velocity exceeding thresholds (teleportation).
2. OutlierVelocityChecker: single-frame velocity spike relative to neighbors (retargeting artifacts).

Rule:
  - Compute per-frame root translation velocity delta[t] = trans[t+1] - trans[t].
  - TransVelocity: if max(|delta[:,axis]|) > threshold for any axis, flag.
  - OutlierVelocity: vel_norm[t] / mean(vel_norm[t-1], vel_norm[t+1]) > sigma, flag.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .base_checker import BaseQualityChecker, CheckResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Per-axis translation velocity thresholds (meters per frame)
DEFAULT_THRESHOLD_X = 1.0
DEFAULT_THRESHOLD_Y = 1.0
DEFAULT_THRESHOLD_Z = 1.0

# Outlier velocity: ratio of current frame velocity to neighbor mean
DEFAULT_OUTLIER_SIGMA = 5.0

MIN_FRAMES_REQUIRED = 4  # Need at least 4 frames for outlier check (vel[1:-1])


class TranslationVelocityChecker(BaseQualityChecker):
    """Detects root translation velocity spikes and outlier velocity.
    Does not require a body model."""

    name = "translation_velocity"

    def __init__(
        self,
        body_model=None,
        device: str = "cpu",
        threshold_x: float = DEFAULT_THRESHOLD_X,
        threshold_y: float = DEFAULT_THRESHOLD_Y,
        threshold_z: float = DEFAULT_THRESHOLD_Z,
        outlier_sigma: float = DEFAULT_OUTLIER_SIGMA,
    ) -> None:
        super().__init__(body_model=body_model, device=device)
        self.threshold_x = threshold_x
        self.threshold_y = threshold_y
        self.threshold_z = threshold_z
        self.outlier_sigma = outlier_sigma

    def get_required_keys(self) -> list:
        return ["trans"]

    def check(self, motion: Union[Dict, str, Path]) -> CheckResult:
        if isinstance(motion, (str, Path)):
            data = self.load_motion(motion)
        else:
            data = dict(motion)
            if "transl" in data and "trans" not in data:
                data["trans"] = data["transl"]

        if "trans" not in data:
            return CheckResult(
                is_valid=False,
                invalid_reason="Missing trans",
                invalid_mask=None,
                details={"has_spike": False, "has_outlier": False, "reason": "Missing trans"},
            )

        trans = np.asarray(data["trans"])
        if trans.ndim != 2 or trans.shape[1] < 3:
            return CheckResult(
                is_valid=False,
                invalid_reason="Invalid trans shape",
                invalid_mask=None,
                details={"has_spike": False, "has_outlier": False, "reason": "Invalid trans shape"},
            )

        if len(trans) < MIN_FRAMES_REQUIRED:
            return CheckResult(
                is_valid=True,
                invalid_reason="Too short",
                invalid_mask=None,
                details={"has_spike": False, "has_outlier": False, "reason": "Too short"},
            )

        # --- Per-axis velocity check ---
        delta = trans[1:] - trans[:-1]  # (F-1, 3)
        max_vel_x = float(np.max(np.abs(delta[:, 0])))
        max_vel_y = float(np.max(np.abs(delta[:, 1])))
        max_vel_z = float(np.max(np.abs(delta[:, 2])))

        velocity_violations: List[str] = []
        if max_vel_x > self.threshold_x:
            velocity_violations.append(f"x={max_vel_x:.3f}>{self.threshold_x:.1f}")
        if max_vel_y > self.threshold_y:
            velocity_violations.append(f"y={max_vel_y:.3f}>{self.threshold_y:.1f}")
        if max_vel_z > self.threshold_z:
            velocity_violations.append(f"z={max_vel_z:.3f}>{self.threshold_z:.1f}")

        # --- Outlier velocity check ---
        vel_norm = np.linalg.norm(delta, axis=1)  # (F-1,)
        has_outlier = False
        outlier_max_ratio = 0.0
        outlier_frame = -1
        if len(vel_norm) >= 3:
            vel_left = vel_norm[:-2]
            vel_right = vel_norm[2:]
            vel_mean = (vel_left + vel_right) / 2.0
            vel_mid = vel_norm[1:-1]
            # Avoid division by zero: clamp denominator
            ratios = vel_mid / np.clip(vel_mean, 1e-2, None)
            outlier_max_ratio = float(np.max(ratios))
            if outlier_max_ratio > self.outlier_sigma:
                has_outlier = True
                outlier_frame = int(np.argmax(ratios)) + 1  # offset by 1 due to slicing

        has_spike = len(velocity_violations) > 0

        if not has_spike and not has_outlier:
            return CheckResult(
                is_valid=True,
                invalid_reason="No translation velocity issue detected",
                invalid_mask=None,
                details={
                    "has_spike": False,
                    "has_outlier": False,
                    "max_vel_x": max_vel_x,
                    "max_vel_y": max_vel_y,
                    "max_vel_z": max_vel_z,
                    "outlier_max_ratio": outlier_max_ratio,
                    "reason": "No translation velocity issue detected",
                },
            )

        reasons = []
        if has_spike:
            reasons.append(f"Trans velocity spike: {', '.join(velocity_violations)}")
        if has_outlier:
            reasons.append(
                f"Outlier velocity at frame {outlier_frame} "
                f"(ratio={outlier_max_ratio:.1f}>{self.outlier_sigma:.1f})"
            )
        reason = "; ".join(reasons)

        details = {
            "has_spike": has_spike,
            "has_outlier": has_outlier,
            "velocity_violations": velocity_violations,
            "max_vel_x": max_vel_x,
            "max_vel_y": max_vel_y,
            "max_vel_z": max_vel_z,
            "outlier_max_ratio": outlier_max_ratio,
            "outlier_frame": outlier_frame,
            "reason": reason,
        }
        return CheckResult(
            is_valid=False,
            invalid_reason=reason,
            invalid_mask=None,
            details=details,
        )


def detect_translation_velocity(data: Dict, **kwargs) -> Dict:
    """Legacy API. Returns dict with keys: has_spike, has_outlier, reason."""
    checker = TranslationVelocityChecker(**kwargs)
    result = checker.check(data)
    details = result.get("details") or {}
    return {
        "has_spike": details.get("has_spike", False),
        "has_outlier": details.get("has_outlier", False),
        "velocity_violations": details.get("velocity_violations", []),
        "outlier_max_ratio": details.get("outlier_max_ratio", 0.0),
        "reason": details.get("reason", result.get("invalid_reason", "")),
    }
