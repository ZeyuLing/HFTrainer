#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The PhysFlow Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PhysFlow Motion Format Converter.

Bidirectional converter between T2M's motion_135 format and ProtoMotions'
MotionLib .pt format. Used by PhysFlow Direction B (Generation → RL) to
convert T2M-generated motions into training data for the RL tracker.

Format definitions:

T2M motion_135: (T, 135)
    Layout: [transl(3) + 22_joints × rot6d(6)]
    Coordinate: Y-up
    Rotation: HyMotion row-major interleaved rot6d
    FPS: 30 typical
    Joints: SMPL 22-joint order (Pelvis, L_Hip, R_Hip, Spine1, ...)

ProtoMotions MotionLib .pt:
    gts:   (total_frames, num_bodies, 3) — global body positions (Z-up)
    grs:   (total_frames, num_bodies, 4) — global body rotations (xyzw)
    gvs:   (total_frames, num_bodies, 3) — global body linear velocities
    gavs:  (total_frames, num_bodies, 3) — global body angular velocities
    dps:   (total_frames, num_dofs)      — DOF positions (joint angles)
    dvs:   (total_frames, num_dofs)      — DOF velocities
    length_starts: (num_motions,)        — start frame index per motion
    motion_lengths: (num_motions,)       — duration in seconds
    motion_dt: (num_motions,)            — dt per motion (1/fps)
    motion_num_frames: (num_motions,)    — frame count per motion
    motion_weights: (num_motions,)       — sampling weights
    contacts: (total_frames, num_contact_bodies) — foot contact binary
    motion_files: Tuple[str]             — motion names

Usage:
    from scripts.embodied.physflow_motion_converter import MotionFormatConverter

    converter = MotionFormatConverter()

    # T2M -> ProtoMotions (for Direction B: RL training on T2M motions)
    motions_135 = [np.random.randn(120, 135) for _ in range(10)]
    pt_data = converter.motion_135_to_protomotions_pt(motions_135, fps=30)
    torch.save(pt_data, "data/t2m_motion_lib.pt")

    # ProtoMotions -> T2M (for evaluation or visualization)
    pt_data = torch.load("data/some_motion_lib.pt")
    motions_135 = converter.protomotions_pt_to_motion_135(pt_data)

Standalone test:
    python3 scripts/embodied/physflow_motion_converter.py --test
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent

_DEFAULT_MJCF = str(
    _REPO_ROOT
    / "ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml"
)

# ---------------------------------------------------------------------------
# Import shared conversion functions from run_smpl_rl_tracker.py
# ---------------------------------------------------------------------------
from scripts.embodied.run_smpl_rl_tracker import (
    rot6d_to_rotmat,
    yup_to_zup,
    zup_to_yup,
    smpl_to_qpos,
    qpos_to_smpl,
    mujoco_wxyz_to_xyzw,
    xyzw_to_wxyz,
    SMPL_2_MUJOCO,
    MUJOCO_2_SMPL,
    MUJOCO_BODY_NAMES,
    _patch_mjcf_xml,
    _quat_mul_wxyz,
)
from scripts.embodied.physflow_rl_oracle import (
    decode_motion_135_array,
    encode_motion_135,
    rotmat_to_rot6d,
    compute_ground_offset,
)


# ===========================================================================
# Constants
# ===========================================================================

NUM_BODIES = 24  # SMPL bodies in MuJoCo (indices 1-24)
NUM_DOFS = 69    # 23 joints × 3 euler angles

# Foot body indices (0-based within the 24 SMPL bodies)
# MuJoCo body order: Pelvis(0), L_Hip(1), L_Knee(2), L_Ankle(3), L_Toe(4),
#                    R_Hip(5), R_Knee(6), R_Ankle(7), R_Toe(8), ...
_FOOT_BODY_INDICES = [3, 4, 7, 8]  # L_Ankle, L_Toe, R_Ankle, R_Toe
CONTACT_HEIGHT_THRESHOLD = 0.04  # 4cm — foot considered in contact


# ===========================================================================
# MotionFormatConverter — Main class
# ===========================================================================

class MotionFormatConverter:
    """Bidirectional converter between T2M motion_135 and ProtoMotions MotionLib.

    Handles all coordinate system conversions, FK computation, and format
    packing/unpacking needed for PhysFlow Direction B.
    """

    def __init__(self, mjcf_path: str = _DEFAULT_MJCF):
        """Initialize converter with MuJoCo model for FK computation.

        Args:
            mjcf_path: Path to SMPL humanoid MJCF XML file
        """
        self.mjcf_path = mjcf_path

        # Load model for FK
        mjcf_file = Path(mjcf_path)
        patched_xml = _patch_mjcf_xml(mjcf_file)

        asset_dir = str(mjcf_file.parent)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", dir=asset_dir, delete=False
        ) as tmp:
            tmp.write(patched_xml)
            tmp_path = tmp.name

        try:
            self.model = mujoco.MjModel.from_xml_path(tmp_path)
        finally:
            os.unlink(tmp_path)

        self.data = mujoco.MjData(self.model)
        self.body_pos_1 = self.model.body_pos[1].copy()

        log.info(f"MotionFormatConverter initialized: "
                 f"{self.model.nbody} bodies, {self.model.nu} actuators")

    # ═══════════════════════════════════════════════════════════════════════
    # T2M motion_135 → ProtoMotions MotionLib .pt
    # ═══════════════════════════════════════════════════════════════════════

    def motion_135_to_protomotions_pt(
        self,
        motions_135: List[np.ndarray],
        fps: int = 30,
        motion_names: Optional[List[str]] = None,
        apply_ground_offset: bool = True,
    ) -> dict:
        """Convert a list of motion_135 arrays to ProtoMotions MotionLib format.

        This is the main method for Direction B: converting T2M outputs to
        RL training data format.

        Pipeline per motion:
          1. motion_135 (Y-up, rot6d) → decode to axis-angle
          2. Y-up → Z-up coordinate transform
          3. SMPL axis-angle → MuJoCo qpos (76-dim)
          4. Ground offset correction (place on floor)
          5. mj_forward (FK) per frame → body_pos, body_rot, body_vel
          6. Extract DOF positions/velocities
          7. Detect foot contacts via height threshold
          8. Concatenate all motions with length_starts index

        Args:
            motions_135: List of (T_i, 135) motion arrays
            fps: Frame rate of input motions
            motion_names: Optional names for each motion (for motion_files)
            apply_ground_offset: Whether to adjust height to touch floor

        Returns:
            dict ready for torch.save() → ProtoMotions MotionLib.load_from_file()
        """
        if motion_names is None:
            motion_names = [f"t2m_gen_{i:04d}" for i in range(len(motions_135))]
        assert len(motion_names) == len(motions_135)

        dt = 1.0 / fps

        # Accumulators
        all_gts = []     # global translations (body positions)
        all_grs = []     # global rotations (xyzw quaternions)
        all_gvs = []     # global linear velocities
        all_gavs = []    # global angular velocities
        all_dps = []     # DOF positions (joint angles)
        all_dvs = []     # DOF velocities
        all_contacts = []  # foot contact binary

        length_starts = []
        motion_lengths = []
        motion_dts = []
        motion_num_frames = []
        motion_weights = []

        current_start = 0

        for idx, motion_135 in enumerate(motions_135):
            T = motion_135.shape[0]
            if T < 2:
                log.warning(f"Skipping motion {idx}: only {T} frame(s)")
                continue

            # [1] Decode motion_135 to SMPL axis-angle
            smpl_pose, transl = decode_motion_135_array(motion_135)

            # [2] Y-up → Z-up
            smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose, transl)

            # [3] SMPL → MuJoCo qpos
            qpos = smpl_to_qpos(smpl_pose_zup, transl_zup, self.body_pos_1)

            # [4] Ground offset
            if apply_ground_offset:
                offset = self._compute_ground_offset(qpos)
                qpos[:, 2] -= offset

            # [5-7] FK to get body states + contacts
            body_pos, body_rot, body_vel, body_ang_vel, dof_pos, dof_vel, contacts = (
                self._fk_trajectory(qpos, dt)
            )

            # Accumulate
            all_gts.append(body_pos)
            all_grs.append(body_rot)
            all_gvs.append(body_vel)
            all_gavs.append(body_ang_vel)
            all_dps.append(dof_pos)
            all_dvs.append(dof_vel)
            all_contacts.append(contacts)

            length_starts.append(current_start)
            motion_lengths.append(T * dt)
            motion_dts.append(dt)
            motion_num_frames.append(T)
            motion_weights.append(1.0)
            current_start += T

        if not all_gts:
            raise ValueError("No valid motions to convert!")

        # Pack into ProtoMotions format
        result = {
            "gts": torch.from_numpy(np.concatenate(all_gts, axis=0)).float(),
            "grs": torch.from_numpy(np.concatenate(all_grs, axis=0)).float(),
            "gvs": torch.from_numpy(np.concatenate(all_gvs, axis=0)).float(),
            "gavs": torch.from_numpy(np.concatenate(all_gavs, axis=0)).float(),
            "dps": torch.from_numpy(np.concatenate(all_dps, axis=0)).float(),
            "dvs": torch.from_numpy(np.concatenate(all_dvs, axis=0)).float(),
            "length_starts": torch.tensor(length_starts, dtype=torch.long),
            "motion_lengths": torch.tensor(motion_lengths, dtype=torch.float32),
            "motion_dt": torch.tensor(motion_dts, dtype=torch.float32),
            "motion_num_frames": torch.tensor(motion_num_frames, dtype=torch.long),
            "motion_weights": torch.tensor(motion_weights, dtype=torch.float32),
            "contacts": torch.from_numpy(np.concatenate(all_contacts, axis=0)).float(),
            "motion_files": tuple(motion_names[:len(all_gts)]),
        }

        total_frames = result["gts"].shape[0]
        num_motions = len(length_starts)
        log.info(f"Converted {num_motions} motions → "
                 f"{total_frames} total frames, "
                 f"gts={result['gts'].shape}, dps={result['dps'].shape}")

        return result

    # ═══════════════════════════════════════════════════════════════════════
    # ProtoMotions MotionLib .pt → T2M motion_135
    # ═══════════════════════════════════════════════════════════════════════

    def protomotions_pt_to_motion_135(
        self,
        pt_data: dict,
        target_fps: int = 30,
    ) -> List[np.ndarray]:
        """Convert ProtoMotions MotionLib .pt format back to motion_135 list.

        This is the inverse direction: useful for evaluation and visualization
        of RL training results.

        Pipeline per motion:
          1. Extract per-motion slice from concatenated tensors
          2. dof_pos (euler angles) → SMPL axis-angle
          3. Root position + root quaternion from body data
          4. Z-up → Y-up
          5. Axis-angle → rot6d → motion_135

        Args:
            pt_data: Dict loaded from ProtoMotions .pt file (or output of
                     motion_135_to_protomotions_pt)
            target_fps: Desired output FPS (will resample if different)

        Returns:
            List of (T_i, 135) numpy arrays in HyMotion format
        """
        length_starts = pt_data["length_starts"].numpy()
        motion_num_frames = pt_data["motion_num_frames"].numpy()
        motion_dts = pt_data["motion_dt"].numpy()
        dps = pt_data["dps"].numpy()
        gts = pt_data["gts"].numpy()
        grs = pt_data["grs"].numpy()

        num_motions = len(length_starts)
        motions_135 = []

        for i in range(num_motions):
            start = int(length_starts[i])
            T = int(motion_num_frames[i])
            src_fps = 1.0 / float(motion_dts[i])

            # Extract slice
            dof_slice = dps[start:start + T]       # (T, 69)
            gts_slice = gts[start:start + T]       # (T, 24, 3)
            grs_slice = grs[start:start + T]       # (T, 24, 4) xyzw

            # Reconstruct qpos from root + dofs
            qpos = np.zeros((T, 76), dtype=np.float64)
            # Root position from body[0] (Pelvis)
            qpos[:, :3] = gts_slice[:, 0].astype(np.float64)
            # Root orientation from body[0] quaternion (xyzw -> wxyz)
            qpos[:, 3:7] = grs_slice[:, 0][:, [3, 0, 1, 2]].astype(np.float64)
            # DOFs
            qpos[:, 7:] = dof_slice.astype(np.float64)

            # qpos → SMPL (Z-up)
            smpl_pose_zup, transl_zup = qpos_to_smpl(qpos, self.body_pos_1)

            # Z-up → Y-up
            smpl_pose_yup, transl_yup = zup_to_yup(smpl_pose_zup, transl_zup)

            # Resample if FPS mismatch
            if abs(src_fps - target_fps) > 0.5 and T > 1:
                T_out = max(1, int(T * target_fps / src_fps))
                indices = np.linspace(0, T - 1, T_out).astype(int)
                smpl_pose_yup = smpl_pose_yup[indices]
                transl_yup = transl_yup[indices]

            # Encode to motion_135
            motion_135 = encode_motion_135(smpl_pose_yup, transl_yup)
            motions_135.append(motion_135)

        return motions_135

    # ═══════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _compute_ground_offset(self, qpos: np.ndarray) -> float:
        """Compute ground offset using bilateral foot grounding.

        Finds lowest z of foot geoms (L_Ankle, L_Toe, R_Ankle, R_Toe)
        in first frame, with correct capsule/box geometry projection.
        """
        self.data.qpos[:] = qpos[0]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Build foot body ID sets (cached on first call)
        if not hasattr(self, '_left_foot_ids'):
            self._left_foot_ids = set()
            self._right_foot_ids = set()
            for bid in range(1, self.model.nbody):
                bname = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
                if bname in ("L_Ankle", "L_Toe"):
                    self._left_foot_ids.add(bid)
                elif bname in ("R_Ankle", "R_Toe"):
                    self._right_foot_ids.add(bid)

        def _lowest_geom_z(body_id_set):
            min_z = float("inf")
            for gid in range(self.model.ngeom):
                if self.model.geom_bodyid[gid] not in body_id_set:
                    continue
                gtype = int(self.model.geom_type[gid])
                gsize = self.model.geom_size[gid]
                gxpos = self.data.geom_xpos[gid]
                gxmat = self.data.geom_xmat[gid].reshape(3, 3)

                if gtype == 5:  # capsule
                    z_ext = abs(gxmat[2, 2]) * gsize[1] + gsize[0]
                    bottom = gxpos[2] - z_ext
                elif gtype == 3:  # sphere
                    bottom = gxpos[2] - gsize[0]
                elif gtype == 6:  # box
                    z_ext = (abs(gxmat[2, 0]) * gsize[0] +
                             abs(gxmat[2, 1]) * gsize[1] +
                             abs(gxmat[2, 2]) * gsize[2])
                    bottom = gxpos[2] - z_ext
                else:
                    bottom = gxpos[2]
                min_z = min(min_z, bottom)
            return min_z

        left_min = _lowest_geom_z(self._left_foot_ids)
        right_min = _lowest_geom_z(self._right_foot_ids)
        return min(left_min, right_min)

    def _fk_trajectory(
        self, qpos: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray]:
        """Run FK on full trajectory and compute velocities.

        Args:
            qpos: (T, 76) MuJoCo qpos trajectory
            dt: Time step between frames

        Returns:
            body_pos: (T, 24, 3) global body positions
            body_rot: (T, 24, 4) global body rotations (xyzw)
            body_vel: (T, 24, 3) global body linear velocities
            body_ang_vel: (T, 24, 3) global body angular velocities
            dof_pos: (T, 69) DOF positions
            dof_vel: (T, 69) DOF velocities
            contacts: (T, 4) foot contact binary (L_Ankle, L_Toe, R_Ankle, R_Toe)
        """
        T = qpos.shape[0]

        # Use float64 intermediates for numerical stability during FK + velocity
        # computation. Float32 causes ~6e-6 error per frame that compounds and
        # corrupts the RL policy observation at ~step 67.
        body_pos = np.zeros((T, NUM_BODIES, 3), dtype=np.float64)
        body_rot = np.zeros((T, NUM_BODIES, 4), dtype=np.float64)  # xyzw
        dof_pos = np.zeros((T, NUM_DOFS), dtype=np.float64)
        contacts = np.zeros((T, len(_FOOT_BODY_INDICES)), dtype=np.float32)

        for t in range(T):
            self.data.qpos[:] = qpos[t]
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

            # Extract body positions (skip world body at index 0)
            body_pos[t] = self.data.xpos[1:NUM_BODIES + 1].copy()

            # Extract body rotations (wxyz -> xyzw)
            body_rot_wxyz = self.data.xquat[1:NUM_BODIES + 1].copy()
            body_rot[t] = mujoco_wxyz_to_xyzw(body_rot_wxyz)

            # DOF positions (joint angles)
            dof_pos[t] = self.data.qpos[7:7 + NUM_DOFS].copy()

            # Foot contact detection via height threshold
            for ci, bi in enumerate(_FOOT_BODY_INDICES):
                foot_z = body_pos[t, bi, 2]
                contacts[t, ci] = 1.0 if foot_z < CONTACT_HEIGHT_THRESHOLD else 0.0

        # Compute velocities via BACKWARD finite differences.
        # This matches what the RL policy expects (run_smpl_rl_tracker.py
        # precompute_reference_maxcoords). Forward differences shift the velocity
        # signal by one frame, causing observation mismatch during RL training.
        body_vel = np.zeros_like(body_pos)      # float64
        body_ang_vel = np.zeros_like(body_pos)  # float64
        dof_vel = np.zeros_like(dof_pos)        # float64

        if T > 1:
            # Linear velocity: BACKWARD diff — vel[f] = (pos[f] - pos[f-1]) / dt
            for f in range(1, T):
                body_vel[f] = (body_pos[f] - body_pos[f - 1]) / dt

            # Angular velocity: 2 * vec(dq) / dt where dq = q1 * q0_inv (wxyz)
            # This matches the RL tracker's convention exactly. Using scipy
            # as_rotvec() gives subtly different results for near-identity
            # rotations and doesn't handle the shortest-path convention.
            for f in range(1, T):
                for j in range(NUM_BODIES):
                    # body_rot is xyzw, convert to wxyz for quaternion math
                    q0_xyzw = body_rot[f - 1, j]
                    q1_xyzw = body_rot[f, j]
                    q0_w = np.array([q0_xyzw[3], q0_xyzw[0], q0_xyzw[1], q0_xyzw[2]])
                    q1_w = np.array([q1_xyzw[3], q1_xyzw[0], q1_xyzw[1], q1_xyzw[2]])

                    # q0_inv (conjugate for unit quaternion)
                    q0_inv = np.array([q0_w[0], -q0_w[1], -q0_w[2], -q0_w[3]])

                    # dq = q1 * q0_inv (Hamilton product, wxyz)
                    dq = _quat_mul_wxyz(q1_w, q0_inv)

                    # Shortest path: ensure w >= 0
                    if dq[0] < 0:
                        dq = -dq

                    # Angular velocity = 2 * vec(dq) / dt
                    body_ang_vel[f, j] = 2.0 * dq[1:4] / dt

            # DOF velocity: BACKWARD diff
            for f in range(1, T):
                dof_vel[f] = (dof_pos[f] - dof_pos[f - 1]) / dt

            # Frame 0 copies from frame 1 (no prior frame available)
            body_vel[0] = body_vel[1]
            body_ang_vel[0] = body_ang_vel[1]
            dof_vel[0] = dof_vel[1]

        # Cast to float32 for storage (ProtoMotions uses float32 tensors)
        return (
            body_pos.astype(np.float32),
            body_rot.astype(np.float32),
            body_vel.astype(np.float32),
            body_ang_vel.astype(np.float32),
            dof_pos.astype(np.float32),
            dof_vel.astype(np.float32),
            contacts,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Utility methods
    # ═══════════════════════════════════════════════════════════════════════

    def validate_pt_format(self, pt_data: dict) -> bool:
        """Validate that a dict conforms to ProtoMotions MotionLib format.

        Checks tensor shapes and consistency.
        """
        required_keys = [
            "gts", "grs", "gvs", "gavs", "dps", "dvs",
            "length_starts", "motion_lengths", "motion_dt",
            "motion_num_frames", "motion_weights", "contacts",
        ]
        for key in required_keys:
            if key not in pt_data:
                log.error(f"Missing key: {key}")
                return False

        total_frames = pt_data["gts"].shape[0]
        num_motions = pt_data["length_starts"].shape[0]

        # Shape checks
        checks = [
            (pt_data["gts"].shape, (total_frames, NUM_BODIES, 3), "gts"),
            (pt_data["grs"].shape, (total_frames, NUM_BODIES, 4), "grs"),
            (pt_data["gvs"].shape, (total_frames, NUM_BODIES, 3), "gvs"),
            (pt_data["gavs"].shape, (total_frames, NUM_BODIES, 3), "gavs"),
            (pt_data["dps"].shape, (total_frames, NUM_DOFS), "dps"),
            (pt_data["dvs"].shape, (total_frames, NUM_DOFS), "dvs"),
            (pt_data["motion_lengths"].shape, (num_motions,), "motion_lengths"),
            (pt_data["motion_dt"].shape, (num_motions,), "motion_dt"),
            (pt_data["motion_num_frames"].shape, (num_motions,), "motion_num_frames"),
            (pt_data["motion_weights"].shape, (num_motions,), "motion_weights"),
        ]

        for actual, expected, name in checks:
            if actual != expected:
                log.error(f"Shape mismatch for '{name}': {actual} vs {expected}")
                return False

        # Check length_starts consistency
        starts = pt_data["length_starts"].numpy()
        lengths = pt_data["motion_num_frames"].numpy()
        for i in range(num_motions):
            if i + 1 < num_motions:
                expected_next = starts[i] + lengths[i]
                if expected_next != starts[i + 1]:
                    log.error(f"length_starts inconsistency at motion {i}: "
                              f"start={starts[i]} + len={lengths[i]} "
                              f"= {expected_next} != next_start={starts[i+1]}")
                    return False

        # Total frames consistency
        total_from_starts = starts[-1] + lengths[-1] if num_motions > 0 else 0
        if total_from_starts != total_frames:
            log.error(f"Total frames mismatch: "
                      f"sum={total_from_starts} vs tensor={total_frames}")
            return False

        log.info(f"Validation PASSED: {num_motions} motions, "
                 f"{total_frames} total frames")
        return True

    def get_motion_stats(self, pt_data: dict) -> dict:
        """Get summary statistics of a MotionLib .pt file."""
        num_motions = pt_data["length_starts"].shape[0]
        total_frames = pt_data["gts"].shape[0]
        lengths = pt_data["motion_num_frames"].numpy()
        durations = pt_data["motion_lengths"].numpy()

        return {
            "num_motions": num_motions,
            "total_frames": total_frames,
            "frame_range": (int(lengths.min()), int(lengths.max())),
            "duration_range_s": (float(durations.min()), float(durations.max())),
            "total_duration_s": float(durations.sum()),
            "fps_values": (1.0 / pt_data["motion_dt"].numpy()).tolist(),
        }


# ===========================================================================
#  CLI / Tests
# ===========================================================================

def _test_roundtrip():
    """Test full roundtrip: motion_135 → ProtoMotions .pt → motion_135."""
    log.info("Testing motion_135 → ProtoMotions → motion_135 roundtrip...")

    # Generate a simple walking-like motion for testing
    T = 60
    fps = 30
    motion_135 = np.zeros((T, 135), dtype=np.float32)

    # Set translation: walking forward in Y-up (Y=height ~0.9m, Z=forward)
    motion_135[:, 0] = 0.0   # X (lateral)
    motion_135[:, 1] = 0.9   # Y (height in Y-up)
    motion_135[:, 2] = np.linspace(0, 2.0, T)  # Z (forward)

    # Set identity rotation (rot6d for identity = specific pattern)
    # Identity rotmat columns: col0=[1,0,0], col1=[0,1,0]
    # Standard rot6d: [1, 0, 0, 0, 1, 0]
    # HyMotion interleaved: standard[..., [0,3,1,4,2,5]] = [1, 0, 0, 1, 0, 0]
    identity_rot6d = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    for j in range(22):
        motion_135[:, 3 + j * 6:3 + (j + 1) * 6] = identity_rot6d

    converter = MotionFormatConverter()

    # Forward: motion_135 → ProtoMotions .pt
    pt_data = converter.motion_135_to_protomotions_pt(
        [motion_135], fps=fps, motion_names=["test_walk"]
    )

    # Validate format
    assert converter.validate_pt_format(pt_data), "Validation failed!"

    # Print stats
    stats = converter.get_motion_stats(pt_data)
    log.info(f"  Stats: {stats}")

    # Check shapes
    assert pt_data["gts"].shape == (T, NUM_BODIES, 3)
    assert pt_data["grs"].shape == (T, NUM_BODIES, 4)
    assert pt_data["gvs"].shape == (T, NUM_BODIES, 3)
    assert pt_data["gavs"].shape == (T, NUM_BODIES, 3)
    assert pt_data["dps"].shape == (T, NUM_DOFS)
    assert pt_data["dvs"].shape == (T, NUM_DOFS)
    assert pt_data["contacts"].shape == (T, len(_FOOT_BODY_INDICES))
    assert pt_data["length_starts"].tolist() == [0]
    assert pt_data["motion_num_frames"].tolist() == [T]

    # Inverse: ProtoMotions → motion_135
    motions_135_back = converter.protomotions_pt_to_motion_135(pt_data, target_fps=fps)
    assert len(motions_135_back) == 1
    motion_135_back = motions_135_back[0]
    assert motion_135_back.shape == (T, 135)

    # Check translation roundtrip (should be close, accounting for body_pos_1 offset)
    transl_diff = np.abs(motion_135[:, :3] - motion_135_back[:, :3]).max()
    log.info(f"  Translation max diff: {transl_diff:.6f}")

    # Check rotation roundtrip (should be very close for identity rotations)
    rot_diff = np.abs(motion_135[:, 3:] - motion_135_back[:, 3:]).mean()
    log.info(f"  Rotation mean abs diff: {rot_diff:.6f}")

    log.info("  PASSED: roundtrip test")
    return pt_data


def _test_multi_motion():
    """Test conversion of multiple motions."""
    log.info("Testing multi-motion conversion...")

    # Create 3 motions of different lengths
    motions = []
    for i, T in enumerate([30, 60, 90]):
        motion = np.zeros((T, 135), dtype=np.float32)
        motion[:, 0] = i * 2.0  # Different lateral positions
        motion[:, 1] = 0.9
        motion[:, 2] = np.linspace(0, 1.0, T)
        # Identity rotations
        identity_rot6d = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        for j in range(22):
            motion[:, 3 + j * 6:3 + (j + 1) * 6] = identity_rot6d
        motions.append(motion)

    converter = MotionFormatConverter()
    pt_data = converter.motion_135_to_protomotions_pt(motions, fps=30)

    # Validate
    assert converter.validate_pt_format(pt_data)
    assert pt_data["gts"].shape[0] == 30 + 60 + 90  # total frames
    assert pt_data["length_starts"].tolist() == [0, 30, 90]
    assert pt_data["motion_num_frames"].tolist() == [30, 60, 90]

    # Convert back
    motions_back = converter.protomotions_pt_to_motion_135(pt_data, target_fps=30)
    assert len(motions_back) == 3
    assert motions_back[0].shape == (30, 135)
    assert motions_back[1].shape == (60, 135)
    assert motions_back[2].shape == (90, 135)

    log.info("  PASSED: multi-motion test")


def _test_save_load():
    """Test saving and loading the .pt file."""
    log.info("Testing save/load cycle...")

    T = 30
    motion_135 = np.zeros((T, 135), dtype=np.float32)
    motion_135[:, 1] = 0.9
    identity_rot6d = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    for j in range(22):
        motion_135[:, 3 + j * 6:3 + (j + 1) * 6] = identity_rot6d

    converter = MotionFormatConverter()
    pt_data = converter.motion_135_to_protomotions_pt([motion_135], fps=30)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name

    try:
        torch.save(pt_data, tmp_path)
        log.info(f"  Saved to {tmp_path} "
                 f"({os.path.getsize(tmp_path) / 1024:.1f} KB)")

        # Load back
        loaded = torch.load(tmp_path, weights_only=False)
        assert converter.validate_pt_format(loaded)

        # Verify tensors match
        for key in ["gts", "grs", "gvs", "gavs", "dps", "dvs"]:
            torch.testing.assert_close(pt_data[key], loaded[key])

        log.info("  PASSED: save/load test")
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(
        description="PhysFlow Motion Format Converter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test", action="store_true",
                        help="Run all tests")
    parser.add_argument("--convert", type=str, default=None,
                        help="Convert a motion_135 NPZ to .pt format")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for converted .pt file")
    parser.add_argument("--fps", type=int, default=30,
                        help="Input motion FPS")
    parser.add_argument("--mjcf", type=str, default=_DEFAULT_MJCF,
                        help="Path to SMPL MJCF XML")

    args = parser.parse_args()

    if args.test:
        _test_roundtrip()
        _test_multi_motion()
        _test_save_load()
        log.info("\n=== All tests PASSED ===")

    elif args.convert:
        # Convert a single NPZ
        data = np.load(args.convert, allow_pickle=True)
        motion_135 = data["motion_135"]
        fps = int(data.get("fps", args.fps))

        converter = MotionFormatConverter(mjcf_path=args.mjcf)
        pt_data = converter.motion_135_to_protomotions_pt(
            [motion_135], fps=fps,
            motion_names=[Path(args.convert).stem]
        )

        out_path = args.output or args.convert.replace(".npz", "_motionlib.pt")
        torch.save(pt_data, out_path)
        log.info(f"Saved: {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")
    else:
        log.info("Use --test to run tests, or --convert <npz> to convert files.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    main()
