"""Load pre-processed o6dp_1103 motion representation.

The o6dp_1103 format stores absolute-translation + rot6d + root-invariant
joint coordinates (RIC) as a flat NumPy array.

For 52 joints (``joints_num=52``), the layout is 471 dims:
  - ``[0:3]``    abs translation (3)
  - ``[3:9]``    root global rot6d (6)
  - ``[9:315]``  body local rot6d ((52-1)*6=306)
  - ``[315:471]`` RIC joints 3D (52*3=156)

For 22 joints (``joints_num=22``), the layout is 201 dims:
  - ``[0:3]``    abs translation (3)
  - ``[3:9]``    root global rot6d (6)
  - ``[9:135]``  body local rot6d ((22-1)*6=126)
  - ``[135:201]`` RIC joints 3D (22*3=66)

This transform loads the 471-dim npy files and extracts the 22-joint
201-dim subset by taking the first 22 joints of rotation and RIC.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from mmcv import BaseTransform

from hftrainer.registry import TRANSFORMS


def _extract_22j_from_52j(motion_52j: np.ndarray) -> np.ndarray:
    """Extract 22-joint 201-dim representation from 52-joint 471-dim.

    Args:
        motion_52j: (T, 471) array in o6dp_1103 52-joint format.

    Returns:
        (T, 201) array in o6dp_1103 22-joint format.
    """
    T = motion_52j.shape[0]

    # Parse 52-joint layout
    transl = motion_52j[:, 0:3]            # (T, 3)
    root_rot6d = motion_52j[:, 3:9]        # (T, 6)
    body_rot6d_52 = motion_52j[:, 9:315]   # (T, 306) = 51 joints * 6
    ric_52 = motion_52j[:, 315:471]        # (T, 156) = 52 joints * 3

    # Extract first 21 body joints (skip hand joints after index 21)
    body_rot6d_22 = body_rot6d_52[:, :21 * 6]  # (T, 126)
    # Extract first 22 RIC joints
    ric_22 = ric_52[:, :22 * 3]                 # (T, 66)

    # Concatenate: [transl(3), root_rot6d(6), body_rot6d(126), ric(66)] = 201
    return np.concatenate([transl, root_rot6d, body_rot6d_22, ric_22], axis=-1)


@TRANSFORMS.register_module(force=True)
class LoadO6dp(BaseTransform):
    """Load pre-processed o6dp_1103 motion npy files.

    Supports loading both 22-joint (201-dim) and 52-joint (471-dim) npy files.
    When ``joints_num=22`` and the file is 471-dim, automatically extracts
    the 22-joint subset.

    Parameters
    ----------
    key : str
        Key to read the motion path from ``results[f'{key}_path']``
        and write the result to ``results[key]``.
    joints_num : int
        Target number of joints: 22 (body-only, 201-dim) or 52 (with hands, 471-dim).
    transl_aug_prob : float
        Probability of applying Y-axis rotation augmentation.
    transl_aug_yaw_deg : float
        Max yaw rotation range in degrees for augmentation.
    transl_aug_offset_std : tuple
        Std of XZ-plane offset augmentation (Y forced to 0).
    """

    def __init__(
        self,
        key: str = 'motion',
        joints_num: int = 22,
        transl_aug_prob: float = 0.75,
        transl_aug_yaw_deg: float = 180.0,
        transl_aug_offset_std: Tuple[float, float, float] = (1.0, 0.0, 1.0),
    ):
        super().__init__()
        assert joints_num in (22, 52), f"joints_num must be 22 or 52, got {joints_num}"
        self.key = key
        self.joints_num = joints_num
        self.expected_dim = 3 + 6 + (joints_num - 1) * 6 + joints_num * 3
        # 22 joints -> 201, 52 joints -> 471

        self.transl_aug_prob = float(transl_aug_prob)
        self.transl_aug_yaw_deg = float(transl_aug_yaw_deg)
        self.transl_aug_offset_std = np.asarray(transl_aug_offset_std, dtype=np.float32)

    def _sample_augmentation(self):
        """Sample Y-axis rotation and XZ offset augmentation."""
        do_aug = (self.transl_aug_prob > 0.0) and (
            np.random.rand() < self.transl_aug_prob
        )
        if not do_aug:
            return False, 0.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        yaw_deg = float(np.random.uniform(-self.transl_aug_yaw_deg, self.transl_aug_yaw_deg))
        yaw = np.deg2rad(yaw_deg)
        c, s = np.cos(yaw), np.sin(yaw)
        R_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

        sx, _, sz = self.transl_aug_offset_std
        offset = np.array(
            [np.random.normal(0, float(sx)), 0.0, np.random.normal(0, float(sz))],
            dtype=np.float32,
        )
        return True, yaw_deg, R_y, offset

    def _apply_augmentation(
        self,
        motion: np.ndarray,
        R_y: np.ndarray,
        offset: np.ndarray,
    ) -> np.ndarray:
        """Apply Y-axis rotation + XZ offset to o6dp_1103 motion.

        Rotates translation, root rotation, and RIC joint positions.
        Body-relative rotations are unchanged (they are parent-relative).
        """
        T = motion.shape[0]
        J = self.joints_num
        D = self.expected_dim

        motion = motion.copy()

        # 1. Rotate and offset translation
        transl = motion[:, 0:3]  # (T, 3)
        transl = transl @ R_y.T + offset[None, :]
        motion[:, 0:3] = transl

        # 2. Rotate root orientation (rot6d)
        # root_rot6d (6) represents first two columns of rotation matrix
        # For row-major: [R00,R01, R10,R11, R20,R21]
        # R_new = R_y @ R_old
        root6d = motion[:, 3:9].reshape(T, 3, 2)  # row-major: each row is [col0, col1]
        # Reconstruct full rotation from first two columns
        col0 = root6d[:, :, 0]  # (T, 3)
        col1 = root6d[:, :, 1]  # (T, 3)
        col0_new = (R_y[None, :, :] @ col0[:, :, None]).squeeze(-1)
        col1_new = (R_y[None, :, :] @ col1[:, :, None]).squeeze(-1)
        root6d_new = np.stack([col0_new, col1_new], axis=-1).reshape(T, 6)
        motion[:, 3:9] = root6d_new

        # 3. Rotate RIC joint positions
        ric_start = 3 + 6 + (J - 1) * 6
        ric = motion[:, ric_start:ric_start + J * 3].reshape(T, J, 3)
        ric = (R_y[None, None, :, :] @ ric[:, :, :, None]).squeeze(-1)
        motion[:, ric_start:ric_start + J * 3] = ric.reshape(T, J * 3)

        return motion

    def transform(self, results: Dict) -> Dict:
        path = results[f'{self.key}_path']
        if isinstance(path, (list, tuple)):
            raise NotImplementedError("LoadO6dp does not support multi-person loading yet.")

        path = str(path)
        motion = np.load(path).astype(np.float32)  # (T, D)

        # Handle dimension mismatch: extract 22-joint from 52-joint
        if motion.shape[1] == 471 and self.joints_num == 22:
            motion = _extract_22j_from_52j(motion)
        elif motion.shape[1] != self.expected_dim:
            raise ValueError(
                f"LoadO6dp: expected {self.expected_dim}-dim motion, got {motion.shape[1]}-dim "
                f"from {path}. Set joints_num appropriately."
            )

        # Augmentation
        do_aug, yaw_deg, R_y, offset = self._sample_augmentation()
        if do_aug:
            motion = self._apply_augmentation(motion, R_y, offset)

        out = torch.from_numpy(motion)
        if torch.any(torch.isnan(out)):
            raise ValueError(f"NaN in {path} after loading/augmentation.")

        results[self.key] = out  # (T, D)
        results['num_person'] = 1
        results['num_frames'] = int(motion.shape[0])
        results['num_joints'] = self.joints_num
        results['aug_yaw_deg'] = yaw_deg if do_aug else 0.0
        results['aug_offset'] = offset.tolist() if do_aug else [0.0, 0.0, 0.0]
        return results
