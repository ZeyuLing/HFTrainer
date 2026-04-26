"""SMPL 22-joint humanoid → Unitree G1 29-DOF retargeting.

Overview
--------
HyMotion T2M-Lite generates 201-dim (or 135-dim) motion in SMPL format:
  - 22 joints × 6 (rotation_6d row-major) + 3 (translation)

Unitree G1 has 29 DOF (or 23 DOF basic version):
  - Legs: 2 × 6 DOF (hip_pitch/roll/yaw, knee, ankle_pitch/roll)
  - Waist: 3 DOF (yaw, roll, pitch)
  - Arms: 2 × 7 DOF (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)

This module implements per-frame retargeting:
  1. SMPL rotation_6d → rotation matrices → axis-angle (per joint)
  2. Map SMPL 3-DOF joints → G1 multi-DOF actuated joints
  3. Decompose SMPL joint rotation matrices into G1 Euler angle DOFs
  4. Clamp to G1 joint limits
  5. Output: (T, 29) joint angle sequence for G1

Joint Correspondence
--------------------
SMPL uses a parent-child kinematic chain with compound 3D rotations.
G1 uses individual 1-DOF revolute joints arranged in pitch/roll/yaw order.

For each SMPL joint, we decompose the 3x3 rotation matrix into the
corresponding Euler angles of the G1 joint group. The decomposition order
matches the G1's kinematic chain (how the revolute joints are stacked).

| SMPL Joint    | G1 Joints                          | Euler Order |
|---------------|-------------------------------------|-------------|
| L_Hip (1)     | l_hip_yaw, l_hip_roll, l_hip_pitch  | ZXY         |
| R_Hip (2)     | r_hip_yaw, r_hip_roll, r_hip_pitch  | ZXY         |
| L_Knee (4)    | l_knee                              | Y (pitch)   |
| R_Knee (5)    | r_knee                              | Y (pitch)   |
| L_Ankle (7)   | l_ankle_pitch, l_ankle_roll         | YX          |
| R_Ankle (8)   | r_ankle_pitch, r_ankle_roll         | YX          |
| Spine1 (3)    | waist_yaw, waist_roll, waist_pitch  | ZXY         |
| L_Shoulder(16)| l_shoulder_pitch/roll/yaw           | YXZ         |
| R_Shoulder(17)| r_shoulder_pitch/roll/yaw           | YXZ         |
| L_Elbow (18)  | l_elbow                             | Y (pitch)   |
| R_Elbow (19)  | r_elbow                             | Y (pitch)   |
| L_Wrist (20)  | l_wrist_roll/pitch/yaw              | XYZ         |
| R_Wrist (21)  | r_wrist_roll/pitch/yaw              | XYZ         |
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_euler,
    rotation_6d_to_matrix,
)


# ============================================================================
# Constants
# ============================================================================

SMPL_JOINT_NAMES = [
    'Pelvis',      # 0
    'L_Hip',       # 1
    'R_Hip',       # 2
    'Spine1',      # 3
    'L_Knee',      # 4
    'R_Knee',      # 5
    'Spine2',      # 6
    'L_Ankle',     # 7
    'R_Ankle',     # 8
    'Spine3',      # 9
    'L_Foot',      # 10
    'R_Foot',      # 11
    'Neck',        # 12
    'L_Collar',    # 13
    'R_Collar',    # 14
    'Head',        # 15
    'L_Shoulder',  # 16
    'R_Shoulder',  # 17
    'L_Elbow',     # 18
    'R_Elbow',     # 19
    'L_Wrist',     # 20
    'R_Wrist',     # 21
]

# G1 29-DOF joint names in order
G1_JOINT_NAMES = [
    # Left leg (6 DOF)
    'left_hip_pitch_joint',     # 0
    'left_hip_roll_joint',      # 1
    'left_hip_yaw_joint',       # 2
    'left_knee_joint',          # 3
    'left_ankle_pitch_joint',   # 4
    'left_ankle_roll_joint',    # 5
    # Right leg (6 DOF)
    'right_hip_pitch_joint',    # 6
    'right_hip_roll_joint',     # 7
    'right_hip_yaw_joint',      # 8
    'right_knee_joint',         # 9
    'right_ankle_pitch_joint',  # 10
    'right_ankle_roll_joint',   # 11
    # Waist (3 DOF)
    'waist_yaw_joint',          # 12
    'waist_roll_joint',         # 13
    'waist_pitch_joint',        # 14
    # Left arm (7 DOF)
    'left_shoulder_pitch_joint',  # 15
    'left_shoulder_roll_joint',   # 16
    'left_shoulder_yaw_joint',    # 17
    'left_elbow_joint',           # 18
    'left_wrist_roll_joint',      # 19
    'left_wrist_pitch_joint',     # 20
    'left_wrist_yaw_joint',       # 21
    # Right arm (7 DOF)
    'right_shoulder_pitch_joint',  # 22
    'right_shoulder_roll_joint',   # 23
    'right_shoulder_yaw_joint',    # 24
    'right_elbow_joint',           # 25
    'right_wrist_roll_joint',      # 26
    'right_wrist_pitch_joint',     # 27
    'right_wrist_yaw_joint',       # 28
]

# G1 joint limits (radians) from URDF.
# Format: (lower, upper) for each of the 29 DOF.
# Source: unitree g1_29dof.urdf typical values.
G1_JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    # Left leg
    'left_hip_pitch_joint':     (-2.5307, 2.8798),
    'left_hip_roll_joint':      (-0.5236, 2.9671),
    'left_hip_yaw_joint':       (-2.7576, 2.7576),
    'left_knee_joint':          (-0.2618, 2.0944),
    'left_ankle_pitch_joint':   (-0.8727, 0.5236),
    'left_ankle_roll_joint':    (-0.2618, 0.2618),
    # Right leg
    'right_hip_pitch_joint':    (-2.5307, 2.8798),
    'right_hip_roll_joint':     (-2.9671, 0.5236),
    'right_hip_yaw_joint':      (-2.7576, 2.7576),
    'right_knee_joint':         (-0.2618, 2.0944),
    'right_ankle_pitch_joint':  (-0.8727, 0.5236),
    'right_ankle_roll_joint':   (-0.2618, 0.2618),
    # Waist
    'waist_yaw_joint':          (-2.6180, 2.6180),
    'waist_roll_joint':         (-0.5236, 0.5236),
    'waist_pitch_joint':        (-0.5236, 0.5236),
    # Left arm
    'left_shoulder_pitch_joint':  (-3.0892, 2.6927),
    'left_shoulder_roll_joint':   (-1.5882, 2.2515),
    'left_shoulder_yaw_joint':    (-2.6180, 2.6180),
    'left_elbow_joint':           (-1.0472, 2.0944),
    'left_wrist_roll_joint':      (-1.9722, 1.9722),
    'left_wrist_pitch_joint':     (-0.3491, 0.3491),
    'left_wrist_yaw_joint':       (-0.5236, 0.5236),
    # Right arm
    'right_shoulder_pitch_joint': (-3.0892, 2.6927),
    'right_shoulder_roll_joint':  (-2.2515, 1.5882),
    'right_shoulder_yaw_joint':   (-2.6180, 2.6180),
    'right_elbow_joint':          (-1.0472, 2.0944),
    'right_wrist_roll_joint':     (-1.9722, 1.9722),
    'right_wrist_pitch_joint':    (-0.3491, 0.3491),
    'right_wrist_yaw_joint':      (-0.5236, 0.5236),
}


# SMPL joint index → G1 DOF mapping definition.
# Each entry: (smpl_joint_idx, euler_order, g1_joint_indices, axis_selection)
#   euler_order: Euler convention used to decompose the SMPL rotation matrix
#   g1_joint_indices: which G1 DOF indices receive the decomposed angles
#   axis_selection: which euler angles to use (None = all 3, or list of indices)
#
# The "axis calibration offset" handles the rest-pose difference between SMPL
# and G1. SMPL's rest pose is T-pose; G1's rest pose has arms down, legs straight.
# We apply a pre-rotation to align coordinate frames.

_JOINT_MAP = [
    # --- Left Leg ---
    # SMPL L_Hip (1) → G1 left_hip_{pitch, roll, yaw} (indices 0,1,2)
    # G1 hip stack: yaw(Z) → roll(X) → pitch(Y) from pelvis
    # Euler decomposition: ZXY → [yaw, roll, pitch]
    # G1 ordering: [pitch(0), roll(1), yaw(2)] so remap euler [2,1,0]
    {
        'smpl_idx': 1,
        'euler_order': 'ZXY',
        'g1_indices': [0, 1, 2],
        'euler_remap': [2, 1, 0],  # pitch=euler[2], roll=euler[1], yaw=euler[0]
    },
    # SMPL L_Knee (4) → G1 left_knee (3), pitch only
    {
        'smpl_idx': 4,
        'euler_order': 'XYZ',
        'g1_indices': [3],
        'euler_remap': [1],  # Y = pitch
    },
    # SMPL L_Ankle (7) → G1 left_ankle_{pitch, roll} (4, 5)
    {
        'smpl_idx': 7,
        'euler_order': 'YXZ',
        'g1_indices': [4, 5],
        'euler_remap': [0, 1],  # pitch=Y(0), roll=X(1)
    },
    # --- Right Leg ---
    {
        'smpl_idx': 2,
        'euler_order': 'ZXY',
        'g1_indices': [6, 7, 8],
        'euler_remap': [2, 1, 0],
    },
    {
        'smpl_idx': 5,
        'euler_order': 'XYZ',
        'g1_indices': [9],
        'euler_remap': [1],
    },
    {
        'smpl_idx': 8,
        'euler_order': 'YXZ',
        'g1_indices': [10, 11],
        'euler_remap': [0, 1],
    },
    # --- Waist ---
    # SMPL Spine1 (3) → G1 waist_{yaw, roll, pitch} (12, 13, 14)
    {
        'smpl_idx': 3,
        'euler_order': 'ZXY',
        'g1_indices': [12, 13, 14],
        'euler_remap': [0, 1, 2],  # yaw=Z(0), roll=X(1), pitch=Y(2)
    },
    # --- Left Arm ---
    # SMPL L_Shoulder (16) → G1 left_shoulder_{pitch, roll, yaw} (15, 16, 17)
    {
        'smpl_idx': 16,
        'euler_order': 'YXZ',
        'g1_indices': [15, 16, 17],
        'euler_remap': [0, 1, 2],  # pitch=Y(0), roll=X(1), yaw=Z(2)
    },
    {
        'smpl_idx': 18,
        'euler_order': 'XYZ',
        'g1_indices': [18],
        'euler_remap': [1],
    },
    # SMPL L_Wrist (20) → G1 left_wrist_{roll, pitch, yaw} (19, 20, 21)
    {
        'smpl_idx': 20,
        'euler_order': 'XYZ',
        'g1_indices': [19, 20, 21],
        'euler_remap': [0, 1, 2],  # roll=X(0), pitch=Y(1), yaw=Z(2)
    },
    # --- Right Arm ---
    {
        'smpl_idx': 17,
        'euler_order': 'YXZ',
        'g1_indices': [22, 23, 24],
        'euler_remap': [0, 1, 2],
    },
    {
        'smpl_idx': 19,
        'euler_order': 'XYZ',
        'g1_indices': [25],
        'euler_remap': [1],
    },
    {
        'smpl_idx': 21,
        'euler_order': 'XYZ',
        'g1_indices': [26, 27, 28],
        'euler_remap': [0, 1, 2],
    },
]


# SMPL parent chain (standard 22-joint)
SMPL_PARENTS = [
    -1,  # 0: Pelvis (root)
     0,  # 1: L_Hip -> Pelvis
     0,  # 2: R_Hip -> Pelvis
     0,  # 3: Spine1 -> Pelvis
     1,  # 4: L_Knee -> L_Hip
     2,  # 5: R_Knee -> R_Hip
     3,  # 6: Spine2 -> Spine1
     4,  # 7: L_Ankle -> L_Knee
     5,  # 8: R_Ankle -> R_Knee
     6,  # 9: Spine3 -> Spine2
     7,  # 10: L_Foot -> L_Ankle
     8,  # 11: R_Foot -> R_Ankle
     9,  # 12: Neck -> Spine3
     9,  # 13: L_Collar -> Spine3
     9,  # 14: R_Collar -> Spine3
    12,  # 15: Head -> Neck
    13,  # 16: L_Shoulder -> L_Collar
    14,  # 17: R_Shoulder -> R_Collar
    16,  # 18: L_Elbow -> L_Shoulder
    17,  # 19: R_Elbow -> R_Shoulder
    18,  # 20: L_Wrist -> L_Elbow
    19,  # 21: R_Wrist -> R_Elbow
]


# ============================================================================
# Retargeter
# ============================================================================

class SMPLToG1Retargeter:
    """Retarget SMPL 22-joint motion to Unitree G1 29-DOF joint angles.

    This class handles:
      1. Converting HyMotion rot6d (row-major) to rotation matrices
      2. Decomposing each SMPL joint rotation into G1 Euler-angle DOFs
      3. Applying rest-pose calibration offsets
      4. Clamping to G1 hardware joint limits
      5. Extracting root (pelvis) position and orientation for the base

    Usage:
        retargeter = SMPLToG1Retargeter()
        g1_result = retargeter.retarget(rot6d, transl)
        # g1_result['joint_angles']: (T, 29) in radians
        # g1_result['root_pos']: (T, 3)
        # g1_result['root_quat']: (T, 4) wxyz quaternion

    Notes:
        - rot6d is in HyMotion row-major convention.
          Must reorder [0,2,4,1,3,5] to column-major before using
          rotation_6d_to_matrix (which expects column-major).
        - The retargeter works in NumPy for broad compatibility.
        - Limb length differences between SMPL and G1 are handled
          implicitly: we only retarget joint angles (not positions).
          The RL policy in Isaac Gym corrects for kinematic differences.
    """

    def __init__(
        self,
        apply_limits: bool = True,
        rest_pose_calibration: bool = True,
        g1_dof: int = 29,
    ):
        self.apply_limits = apply_limits
        self.rest_pose_calibration = rest_pose_calibration
        self.g1_dof = g1_dof
        self.joint_map = _JOINT_MAP

        # Build joint limit arrays
        self._build_limit_arrays()

        # Precompute rest-pose offset rotations.
        # SMPL T-pose → G1 rest pose:
        #   - Shoulders: SMPL T-pose has arms horizontal (90° abduction).
        #     G1 rest pose has arms roughly down. We subtract ~90° from shoulder roll.
        #   - Legs: both roughly aligned in rest pose, minimal offset needed.
        self._shoulder_offset_l = np.array([0.0, -np.pi/2, 0.0])  # pitch, roll, yaw
        self._shoulder_offset_r = np.array([0.0,  np.pi/2, 0.0])

    def _build_limit_arrays(self):
        """Build (29,) arrays of lower/upper limits."""
        self.lower_limits = np.zeros(self.g1_dof)
        self.upper_limits = np.zeros(self.g1_dof)
        for i, name in enumerate(G1_JOINT_NAMES[:self.g1_dof]):
            lo, hi = G1_JOINT_LIMITS.get(name, (-np.pi, np.pi))
            self.lower_limits[i] = lo
            self.upper_limits[i] = hi

    def retarget(
        self,
        rot6d: np.ndarray,
        transl: np.ndarray,
        fps: float = 30.0,
    ) -> Dict[str, np.ndarray]:
        """Retarget SMPL motion to G1 joint angles.

        Args:
            rot6d: (T, 22, 6) rotation 6D in HyMotion row-major convention.
                   Or (T, 132) which will be reshaped.
            transl: (T, 3) root translation in meters.
            fps: frames per second of the motion.

        Returns:
            Dict with keys:
                joint_angles: (T, 29) G1 joint angles in radians
                root_pos: (T, 3) root position (meters)
                root_orient_quat: (T, 4) root orientation (wxyz quaternion)
                root_orient_euler: (T, 3) root orientation (XYZ Euler radians)
                fps: float
                joint_names: list of G1 joint names
        """
        if rot6d.ndim == 2 and rot6d.shape[-1] == 132:
            rot6d = rot6d.reshape(-1, 22, 6)
        assert rot6d.ndim == 3 and rot6d.shape[1] == 22 and rot6d.shape[2] == 6

        T = rot6d.shape[0]

        # Step 1: Convert row-major rot6d to rotation matrices.
        # HyMotion row-major: [R00,R01,R10,R11,R20,R21]
        # rotation_convert column-major: [R00,R10,R20,R01,R11,R21]
        # Reorder: row[0,2,4,1,3,5] → column
        rot6d_col = rot6d[..., [0, 2, 4, 1, 3, 5]]
        # (T, 22, 6) -> (T, 22, 3, 3) rotation matrices
        rotmats = rotation_6d_to_matrix(rot6d_col.reshape(-1, 6)).reshape(T, 22, 3, 3)

        # Step 2: Extract root orientation
        root_rotmat = rotmats[:, 0]  # (T, 3, 3)
        from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
            matrix_to_quaternion,
        )
        root_quat = matrix_to_quaternion(root_rotmat)  # (T, 4) wxyz
        root_euler = matrix_to_euler(root_rotmat, order='XYZ', deg=False)

        # Step 3: Decompose each mapped joint's rotation matrix to G1 DOFs
        joint_angles = np.zeros((T, self.g1_dof), dtype=np.float64)

        for mapping in self.joint_map:
            smpl_idx = mapping['smpl_idx']
            euler_order = mapping['euler_order']
            g1_indices = mapping['g1_indices']
            euler_remap = mapping['euler_remap']

            # Get local rotation matrix for this SMPL joint
            local_rot = rotmats[:, smpl_idx]  # (T, 3, 3)

            # Decompose to Euler angles
            euler_angles = matrix_to_euler(local_rot, order=euler_order, deg=False)
            # euler_angles: (T, 3)

            # Apply rest-pose calibration for shoulders
            if self.rest_pose_calibration:
                if smpl_idx == 16:  # L_Shoulder
                    euler_angles = euler_angles - self._shoulder_offset_l
                elif smpl_idx == 17:  # R_Shoulder
                    euler_angles = euler_angles - self._shoulder_offset_r

            # Map selected euler angles to G1 DOFs
            # Skip joints beyond the configured DOF count (e.g., wrist joints for 23-DOF)
            for i, g1_idx in enumerate(g1_indices):
                if g1_idx >= self.g1_dof:
                    continue
                euler_component = euler_remap[i]
                joint_angles[:, g1_idx] = euler_angles[:, euler_component]

        # Step 4: Clamp to joint limits
        if self.apply_limits:
            joint_angles = np.clip(joint_angles, self.lower_limits, self.upper_limits)

        return {
            'joint_angles': joint_angles.astype(np.float32),
            'root_pos': transl.astype(np.float32),
            'root_orient_quat': root_quat.astype(np.float32),
            'root_orient_euler': root_euler.astype(np.float32),
            'fps': fps,
            'joint_names': G1_JOINT_NAMES[:self.g1_dof],
            'dof': self.g1_dof,
        }

    def retarget_from_hymotion(
        self,
        motion_135: np.ndarray,
        fps: float = 30.0,
    ) -> Dict[str, np.ndarray]:
        """Retarget from HyMotion 135-dim format directly.

        Args:
            motion_135: (T, 135) raw HyMotion output.
                dims [0:3] = translation, dims [3:135] = 22 joints × 6 rot6d.

        Returns:
            Same as retarget().
        """
        assert motion_135.ndim == 2 and motion_135.shape[-1] == 135
        transl = motion_135[:, 0:3]
        rot6d = motion_135[:, 3:135].reshape(-1, 22, 6)
        return self.retarget(rot6d, transl, fps=fps)

    def retarget_from_hymotion_201(
        self,
        motion_201: np.ndarray,
        fps: float = 30.0,
    ) -> Dict[str, np.ndarray]:
        """Retarget from HyMotion T2M 201-dim format.

        Args:
            motion_201: (T, 201) = [transl(3), rot6d(132), joint_pos(66)]
                We only use transl and rot6d, not joint_pos.

        Returns:
            Same as retarget().
        """
        assert motion_201.ndim == 2 and motion_201.shape[-1] == 201
        transl = motion_201[:, 0:3]
        rot6d = motion_201[:, 3:135].reshape(-1, 22, 6)
        return self.retarget(rot6d, transl, fps=fps)

    def to_asap_pkl(
        self,
        retarget_result: Dict[str, np.ndarray],
        output_path: str,
    ) -> str:
        """Save retargeted motion in ASAP-compatible pickle format.

        The ASAP/HumanoidVerse framework expects motion data as a dict
        containing robot joint positions (q) and root state.

        Args:
            retarget_result: output of retarget() or retarget_from_hymotion()
            output_path: path to save .pkl file

        Returns:
            output_path
        """
        import pickle

        motion_data = {
            'fps': retarget_result['fps'],
            'joint_names': retarget_result['joint_names'],
            'dof': retarget_result['dof'],
            # Robot joint angles per frame
            'dof_pos': retarget_result['joint_angles'],   # (T, 29)
            # Root state
            'root_pos': retarget_result['root_pos'],       # (T, 3)
            'root_orient_quat': retarget_result['root_orient_quat'],  # (T, 4) wxyz
            'root_orient_euler': retarget_result['root_orient_euler'],
            # Velocities (finite difference, needed by ASAP)
            'dof_vel': np.gradient(
                retarget_result['joint_angles'],
                1.0 / retarget_result['fps'],
                axis=0,
            ).astype(np.float32),
            'root_vel': np.gradient(
                retarget_result['root_pos'],
                1.0 / retarget_result['fps'],
                axis=0,
            ).astype(np.float32),
        }

        import os
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(motion_data, f)

        return output_path

    def to_mujoco_qpos(
        self,
        retarget_result: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Convert retargeted motion to MuJoCo qpos format.

        MuJoCo qpos for a floating-base humanoid:
          [root_pos(3), root_quat(4 wxyz), joint_angles(29)] = 36 dims

        Returns:
            (T, 36) qpos array.
        """
        T = retarget_result['joint_angles'].shape[0]
        qpos = np.zeros((T, 3 + 4 + self.g1_dof), dtype=np.float32)
        qpos[:, 0:3] = retarget_result['root_pos']
        qpos[:, 3:7] = retarget_result['root_orient_quat']
        qpos[:, 7:7 + self.g1_dof] = retarget_result['joint_angles']
        return qpos
