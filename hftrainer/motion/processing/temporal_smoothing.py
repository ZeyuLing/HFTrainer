"""Temporal smoothing utilities for SMPL ``motion_135`` and SMPLX dictionaries.

The default parameters mirror HY-Motion-1.0 inference: Gaussian quaternion
smoothing with ``sigma=1.0`` for rotations and Savitzky-Golay smoothing with
``window_length=11, polyorder=5`` for root translation.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from hftrainer.motion.representation.rotation import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)
from hftrainer.motion.skeleton.fk import (
    rot6d_to_rotmat_row_major,
    rotmat_to_rot6d_row_major,
)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _copy_value(v: Any) -> Any:
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
        return v.copy()
    return v


def smooth_motion135_hymotion(
    motion_135: np.ndarray,
    rot_sigma: float = 1.0,
    transl_window: int = 11,
    transl_polyorder: int = 5,
) -> np.ndarray:
    """Apply HY-Motion-style smoothing to ``motion_135``.

    Args:
        motion_135: Array with shape ``(T, 135+)``.  The first 135 dimensions are
            ``[translation(3), 22 * row-major rot6d]``.  Extra channels, if any,
            are copied through unchanged.
        rot_sigma: Quaternion Gaussian smoothing sigma.  ``<=0`` disables
            rotational smoothing.
        transl_window: Savitzky-Golay window for translation.  If the sequence
            is not longer than the window, translation is copied through.
        transl_polyorder: Savitzky-Golay polynomial order.

    Returns:
        Smoothed array with the same shape as ``motion_135``.
    """

    m = np.asarray(motion_135, dtype=np.float32)
    if m.ndim != 2 or m.shape[1] < 135:
        raise ValueError(f"Expected motion_135 shape (T,135+), got {m.shape}")

    from hftrainer.models.motion.hymotion_t2m._smoothing import (
        matrix_to_quaternion,
        quaternion_to_matrix,
        smooth_rotation,
        smooth_with_savgol,
    )

    out = m.copy()
    T = m.shape[0]
    transl = torch.from_numpy(m[:, :3])
    rot6d_row = torch.from_numpy(m[:, 3:135]).reshape(T, 22, 6)

    if rot_sigma > 0:
        rotmat = rot6d_to_rotmat_row_major(rot6d_row)
        quat = matrix_to_quaternion(rotmat).numpy()
        quat_s = smooth_rotation(quat.copy(), sigma=rot_sigma)
        rotmat_s = quaternion_to_matrix(torch.from_numpy(quat_s))
        rot6d_row_s = rotmat_to_rot6d_row_major(rotmat_s).reshape(T, 132)
    else:
        rot6d_row_s = rot6d_row.reshape(T, 132)

    if T > transl_window:
        transl_s = smooth_with_savgol(
            transl,
            window_length=transl_window,
            polyorder=transl_polyorder,
        )
    else:
        transl_s = transl

    out[:, :3] = transl_s.float().numpy()
    out[:, 3:135] = rot6d_row_s.float().numpy()
    return out


def smplx_dict_to_motion135(smplx_dict: Mapping[str, Any]) -> np.ndarray:
    """Convert a SMPLX-style generation dictionary to row-major ``motion_135``."""

    transl = _to_numpy(smplx_dict["transl"]).astype(np.float32)
    T = transl.shape[0]
    global_orient = _to_numpy(smplx_dict["global_orient"]).astype(np.float32).reshape(T, 3)
    body_pose = _to_numpy(smplx_dict["body_pose"]).astype(np.float32).reshape(T, 21, 3)
    aa = np.concatenate([global_orient[:, None], body_pose], axis=1)
    rotmat = axis_angle_to_matrix(aa.reshape(-1, 3)).reshape(T, 22, 3, 3)
    rot6d = rotmat_to_rot6d_row_major(torch.from_numpy(rotmat.astype(np.float32))).numpy()
    return np.concatenate([transl, rot6d.reshape(T, 132)], axis=-1).astype(np.float32)


def motion135_to_smplx_dict(
    motion_135: np.ndarray,
    template: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert row-major ``motion_135`` back to a SMPLX-style dictionary."""

    m = np.asarray(motion_135, dtype=np.float32)
    T = m.shape[0]
    rot6d = torch.from_numpy(m[:, 3:135]).reshape(T, 22, 6)
    rotmat = rot6d_to_rotmat_row_major(rot6d).numpy()
    axis_angle = matrix_to_axis_angle(rotmat.reshape(-1, 3, 3)).astype(np.float32).reshape(T, 22, 3)

    out = {k: _copy_value(v) for k, v in (template or {}).items()}
    out["transl"] = m[:, :3].astype(np.float32)
    out["global_orient"] = axis_angle[:, 0].astype(np.float32)
    out["body_pose"] = axis_angle[:, 1:].reshape(T, 63).astype(np.float32)
    return out


def smooth_smplx_dict_hymotion(
    smplx_dict: Mapping[str, Any],
    rot_sigma: float = 1.0,
    transl_window: int = 11,
    transl_polyorder: int = 5,
) -> dict[str, Any]:
    """Apply HY-Motion-style smoothing to a generated SMPLX dictionary."""

    motion_135 = smplx_dict_to_motion135(smplx_dict)
    smooth = smooth_motion135_hymotion(
        motion_135,
        rot_sigma=rot_sigma,
        transl_window=transl_window,
        transl_polyorder=transl_polyorder,
    )
    return motion135_to_smplx_dict(smooth, template=smplx_dict)


__all__ = [
    "motion135_to_smplx_dict",
    "smooth_motion135_hymotion",
    "smooth_smplx_dict_hymotion",
    "smplx_dict_to_motion135",
]
