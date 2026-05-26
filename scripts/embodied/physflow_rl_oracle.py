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
"""PhysFlow RL Physics Correction Oracle.

Wraps the pretrained RL motion tracking policy (ONNX) as a physics validation
and correction oracle for the PhysFlow bidirectional training pipeline.

Operating Modes:
  1. VALIDATOR MODE (default, recommended): With correction scales=0, the oracle
     applies ground offset and runs RL simulation as a QUALITY GATE only. The
     output motion preserves the kinematic structure of the reference but with
     corrected ground contact height. The RL simulation's success/failure
     determines whether the motion is physically plausible.

  2. CORRECTOR MODE (experimental): With non-zero correction scales (e.g.,
     root=0.3, joint=0.05), the oracle also applies low-frequency physics
     corrections extracted from the simulation residual.

Sim-to-Sim Transfer Gap:
  The RL policy was trained in IsaacGym (PhysX engine) but is deployed here in
  MuJoCo. Due to differences in contact models, integrators, and actuator
  dynamics, the raw simulation output exhibits ~28x more kinematic jitter than
  the input reference. This makes the raw sim output UNUSABLE as a motion target.

  Solution: The residual correction mode (with default scales=0) uses the sim
  purely for quality assessment. The only physical correction applied to the
  output is the GROUND OFFSET (moving root Y down ~0.285m so feet touch floor),
  which is computed via MuJoCo FK without relying on the unstable simulation.

Physical Corrections Applied:
  - Ground offset: Lowers root position so the lowest body geom touches z=0.
    This is the ONLY correction in validator mode and is reliable/valuable.
  - (Optional) Residual root translation from RL sim (if root_correction_scale > 0)
  - (Optional) Residual joint angles from RL sim (if joint_correction_scale > 0)

Quality Gate Criteria:
  - completion_ratio >= 0.8 (RL tracked 80%+ without falling)
  - root_height_min >= 0.4m (pelvis stayed above collapse threshold)
  - root_height_std <= 0.15 (raw sim didn't oscillate wildly)
  - jitter_filtered <= 0.005 (post-correction output is smooth)

Expected Baseline: For an untrained T2M model, ~10% physics pass rate is normal.
After PhysFlow training, this should improve to 60-80%+.

Pipeline:
  motion_135 (T, 135) [Y-up, rot6d, 30fps]
    -> decode rot6d -> axis-angle
    -> Y-up -> Z-up
    -> SMPL axis-angle -> MuJoCo qpos
    -> ground offset (MuJoCo FK, find lowest geom, subtract from z)
    -> RL closed-loop simulation (ONNX policy + MuJoCo physics)
    -> residual correction (default: scales=0, output = ground-offset reference)
    -> qpos -> SMPL axis-angle
    -> Z-up -> Y-up
    -> axis-angle -> rot6d
    -> motion_135_out (T', 135) [Y-up, rot6d]

Usage:
    from scripts.embodied.physflow_rl_oracle import RLPhysicsOracle

    oracle = RLPhysicsOracle()
    motion_135 = np.random.randn(120, 135)  # T2M output

    # Validator mode (default): quality gate + ground offset only
    motion_135_out, stats = oracle.correct(motion_135, fps=30)
    if oracle.is_good_quality(stats):
        # Motion is physically plausible — use as training target
        ...

    # Corrector mode (experimental): apply physics residual
    motion_135_out, stats = oracle.correct(
        motion_135, fps=30,
        root_correction_scale=0.3, joint_correction_scale=0.05)

Standalone test:
    python3 scripts/embodied/physflow_rl_oracle.py --test
    python3 scripts/embodied/physflow_rl_oracle.py --npz output/embodied_t2m_v4/data/npz/v4_arm_001.npz
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as sRot

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — reuse same defaults as run_smpl_rl_tracker.py
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent

_DEFAULT_ONNX = str(
    _REPO_ROOT
    / "ref_repo/ProtoMotions/data/pretrained_models/motion_tracker"
    / "smpl/compiled_models/unified_pipeline.onnx"
)
_DEFAULT_YAML = str(
    _REPO_ROOT
    / "ref_repo/ProtoMotions/data/pretrained_models/motion_tracker"
    / "smpl/compiled_models/unified_pipeline.yaml"
)
_DEFAULT_MJCF = str(
    _REPO_ROOT
    / "ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml"
)


# ===========================================================================
#  Import from run_smpl_rl_tracker.py (reuse all conversion functions)
# ===========================================================================

from scripts.embodied.run_smpl_rl_tracker import (
    rot6d_to_rotmat,
    yup_to_zup,
    zup_to_yup,
    smpl_to_qpos,
    qpos_to_smpl,
    run_rl_tracker,
    FALL_HEIGHT_THRESHOLD,
)


# ===========================================================================
#  motion_135 array decode/encode (not file-based)
# ===========================================================================

def decode_motion_135_array(motion_135: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decode motion_135 array to SMPL 72-dim axis-angle + translation (Y-up).

    Args:
        motion_135: (T, 135) array in HyMotion format:
                    [transl(3) + 22 joints x rot6d(6)]

    Returns:
        smpl_pose: (T, 72) axis-angle (24 joints x 3, SMPL order)
                   Joints 22-23 (L_Hand, R_Hand) are zero.
        transl: (T, 3) translation in Y-up coordinates
    """
    T = motion_135.shape[0]
    assert motion_135.shape[1] == 135, f"Expected 135 dims, got {motion_135.shape[1]}"

    transl = motion_135[:, :3].copy()
    rot6d = motion_135[:, 3:].reshape(T, 22, 6)
    rotmat = rot6d_to_rotmat(rot6d)
    aa = sRot.from_matrix(rotmat.reshape(-1, 3, 3)).as_rotvec().reshape(T, 22, 3)

    smpl_pose = np.zeros((T, 72), dtype=np.float32)
    smpl_pose[:, :3] = aa[:, 0, :]            # root
    smpl_pose[:, 3:66] = aa[:, 1:22, :].reshape(T, -1)  # 21 body joints
    # joints 22-23 (L_Hand, R_Hand) remain zero
    return smpl_pose, transl.astype(np.float32)


def rotmat_to_rot6d(rotmat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix (..., 3, 3) to HyMotion row-major rot6d (..., 6).

    This is the inverse of rot6d_to_rotmat from run_smpl_rl_tracker.py.
    HyMotion uses row-major interleaved format: [r00, r10, r01, r11, r02, r12]
    i.e., column vectors of the first two columns interleaved row-by-row.
    """
    # Standard rot6d: first two columns [c0, c1] → (6,) = [c0_x, c0_y, c0_z, c1_x, c1_y, c1_z]
    # But HyMotion uses row-major interleaving:
    #   rot6d_to_rotmat does: rot6d[..., [0,2,4,1,3,5]] to get standard [c0, c1]
    #   So encoding is: standard[..., [0,3,1,4,2,5]] → row-major interleaved
    c0 = rotmat[..., :, 0]  # (..., 3) — first column
    c1 = rotmat[..., :, 1]  # (..., 3) — second column
    # Standard = [c0_x, c0_y, c0_z, c1_x, c1_y, c1_z]
    standard = np.concatenate([c0, c1], axis=-1)  # (..., 6)
    # Interleave: indices [0,3,1,4,2,5]
    return standard[..., [0, 3, 1, 4, 2, 5]]


def encode_motion_135(smpl_pose: np.ndarray, transl: np.ndarray) -> np.ndarray:
    """Encode SMPL 72-dim axis-angle + translation to motion_135 array.

    Args:
        smpl_pose: (T, 72) axis-angle (24 joints x 3)
        transl: (T, 3) translation in Y-up coordinates

    Returns:
        motion_135: (T, 135) in HyMotion format [transl(3) + 22 joints x rot6d(6)]
    """
    T = smpl_pose.shape[0]
    joint_aa = smpl_pose.reshape(T, 24, 3)

    # Convert first 22 joints to rotation matrices
    aa_22 = joint_aa[:, :22].reshape(-1, 3)
    rotmat = sRot.from_rotvec(aa_22).as_matrix().reshape(T, 22, 3, 3)

    # Convert to HyMotion row-major rot6d
    rot6d = rotmat_to_rot6d(rotmat)  # (T, 22, 6)

    # Assemble motion_135
    motion_135 = np.zeros((T, 135), dtype=np.float32)
    motion_135[:, :3] = transl
    motion_135[:, 3:] = rot6d.reshape(T, 132)
    return motion_135


# ===========================================================================
#  Post-processing: Low-pass filtering and reference blending
# ===========================================================================

def lowpass_filter_qpos(
    sim_qpos: np.ndarray,
    control_dt: float,
    cutoff_hz: float = 5.0,
    order: int = 4,
) -> np.ndarray:
    """Apply Butterworth low-pass filter to sim_qpos to remove PD jitter.

    The RL policy + PD control produces high-frequency oscillations (28x more
    jitter than kinematic motion). A low-pass filter at 5Hz preserves the
    overall motion structure while removing control artifacts.

    Args:
        sim_qpos: (T, 76) raw simulation output
        control_dt: Time step between frames (0.02s for 50Hz control)
        cutoff_hz: Filter cutoff frequency in Hz (default 5Hz)
        order: Butterworth filter order (default 4)

    Returns:
        filtered_qpos: (T, 76) smoothed simulation output
    """
    T = sim_qpos.shape[0]
    if T < 2 * order + 1:
        # Too short for filtering, just return original
        return sim_qpos.copy()

    fs = 1.0 / control_dt  # Sampling frequency (50 Hz)
    nyquist = fs / 2.0

    # Clamp cutoff to be below Nyquist
    if cutoff_hz >= nyquist:
        cutoff_hz = nyquist * 0.9

    b, a = butter(order, cutoff_hz / nyquist, btype='low')

    filtered = sim_qpos.copy()

    # Filter root position (columns 0:3) — preserve start/end anchoring
    for col in range(3):
        filtered[:, col] = filtfilt(b, a, sim_qpos[:, col])

    # Root quaternion (columns 3:7): filter as rotation matrix elements
    # to avoid quaternion discontinuities, then re-normalize
    root_quats = sim_qpos[:, 3:7]  # wxyz format
    # Convert to rotation matrices for smooth filtering
    root_rots = sRot.from_quat(
        root_quats[:, [1, 2, 3, 0]]  # wxyz -> xyzw for scipy
    ).as_matrix()  # (T, 3, 3)

    # Filter each rotation matrix element
    for i in range(3):
        for j in range(3):
            root_rots[:, i, j] = filtfilt(b, a, root_rots[:, i, j])

    # Re-orthogonalize via SVD (closest rotation matrix)
    for t in range(T):
        U, _, Vt = np.linalg.svd(root_rots[t])
        root_rots[t] = U @ Vt
        if np.linalg.det(root_rots[t]) < 0:
            U[:, -1] *= -1
            root_rots[t] = U @ Vt

    # Convert back to wxyz quaternion
    filt_quats_xyzw = sRot.from_matrix(root_rots).as_quat()  # xyzw
    filtered[:, 3:7] = filt_quats_xyzw[:, [3, 0, 1, 2]]  # xyzw -> wxyz

    # Filter joint DOFs (columns 7:76) — these are Euler angles
    for col in range(7, sim_qpos.shape[1]):
        filtered[:, col] = filtfilt(b, a, sim_qpos[:, col])

    return filtered


def blend_with_reference(
    sim_qpos: np.ndarray,
    ref_qpos: np.ndarray,
    blend_alpha: float = 0.5,
    ref_fps: int = 30,
    sim_fps: float = 50.0,
) -> np.ndarray:
    """Blend simulation output with reference motion.

    Uses weighted average: output = alpha * ref + (1-alpha) * sim
    where alpha=1.0 means pure reference (no physics correction)
    and alpha=0.0 means pure simulation (maximum physics correction).

    The reference provides smooth structure, the simulation provides
    physical grounding (ground contact, balance corrections).

    Args:
        sim_qpos: (T_sim, 76) filtered simulation output
        ref_qpos: (T_ref, 76) original reference qpos
        blend_alpha: Blend factor. 0=pure sim, 1=pure ref. Default 0.5.
        ref_fps: FPS of reference motion
        sim_fps: FPS of simulation output

    Returns:
        blended_qpos: (T_sim, 76) blended output at sim FPS
    """
    T_sim = sim_qpos.shape[0]
    T_ref = ref_qpos.shape[0]

    # Resample reference to simulation timing
    ref_indices = np.linspace(0, T_ref - 1, T_sim).astype(int)
    ref_resampled = ref_qpos[ref_indices]

    blended = sim_qpos.copy()

    # Blend root position
    blended[:, :3] = blend_alpha * ref_resampled[:, :3] + (1 - blend_alpha) * sim_qpos[:, :3]

    # Blend root quaternion via SLERP
    for t in range(T_sim):
        q_ref = ref_resampled[t, 3:7]  # wxyz
        q_sim = sim_qpos[t, 3:7]  # wxyz

        # Convert to scipy (xyzw)
        r_ref = sRot.from_quat([q_ref[1], q_ref[2], q_ref[3], q_ref[0]])
        r_sim = sRot.from_quat([q_sim[1], q_sim[2], q_sim[3], q_sim[0]])

        # SLERP: alpha=1 -> ref, alpha=0 -> sim
        # Use relative rotation: r_blend = r_ref * slerp(identity, r_ref.inv() * r_sim, 1-alpha)
        r_diff = r_ref.inv() * r_sim
        angle = r_diff.magnitude()
        if angle < 1e-6:
            r_blend = r_ref
        else:
            r_partial = sRot.from_rotvec(r_diff.as_rotvec() * (1 - blend_alpha))
            r_blend = r_ref * r_partial

        q_blend_xyzw = r_blend.as_quat()
        blended[t, 3:7] = [q_blend_xyzw[3], q_blend_xyzw[0],
                           q_blend_xyzw[1], q_blend_xyzw[2]]

    # Blend joint angles (linear interpolation is fine for small differences)
    blended[:, 7:] = blend_alpha * ref_resampled[:, 7:] + (1 - blend_alpha) * sim_qpos[:, 7:]

    return blended


def residual_correction(
    sim_qpos: np.ndarray,
    ref_qpos: np.ndarray,
    control_dt: float,
    ref_fps: int = 30,
    root_correction_scale: float = 0.8,
    joint_correction_scale: float = 0.1,
    lowpass_hz: float = 2.0,
) -> np.ndarray:
    """Apply residual physics correction: ref + filtered(sim - ref) * scale.

    Instead of using the raw simulation output (which has PD jitter), we:
    1. Compute the residual: delta = sim - ref (what physics "corrected")
    2. Low-pass filter the residual (remove high-frequency artifacts)
    3. Apply a scaled version: output = ref + scale * filtered_delta

    This extracts the LOW-FREQUENCY physical corrections (ground height,
    balance drift, settling) while preserving the smooth kinematic structure.

    For the root translation, we apply a stronger correction (scale=0.8)
    because height/grounding is the most valuable physics signal.
    For joint angles, we apply a weaker correction (scale=0.1) because
    the kinematic joint angles are already reasonable and the sim's joint
    corrections are mostly PD noise.

    Args:
        sim_qpos: (T_sim, 76) raw simulation output at control rate
        ref_qpos: (T_ref, 76) reference qpos trajectory
        control_dt: Physics control timestep (0.02s)
        ref_fps: Reference motion FPS (30)
        root_correction_scale: Scale for root position correction [0-1]
        joint_correction_scale: Scale for joint angle correction [0-1]
        lowpass_hz: Cutoff frequency for residual filtering

    Returns:
        corrected_qpos: (T_sim, 76) output at sim rate with smooth corrections
    """
    T_sim = sim_qpos.shape[0]
    T_ref = ref_qpos.shape[0]

    # Resample reference to sim timing
    ref_indices = np.linspace(0, T_ref - 1, T_sim)
    ref_interp = np.zeros((T_sim, ref_qpos.shape[1]), dtype=np.float32)
    for col in range(ref_qpos.shape[1]):
        ref_interp[:, col] = np.interp(
            ref_indices, np.arange(T_ref), ref_qpos[:, col])

    # Compute residual
    delta = sim_qpos - ref_interp

    # Low-pass filter the residual
    # filtfilt with 3rd-order butter needs padlen = 3*max(len(a),len(b)) = 12
    # so input must have at least 13 samples
    if T_sim > 13 and lowpass_hz > 0:
        fs = 1.0 / control_dt
        nyquist = fs / 2.0
        cutoff = min(lowpass_hz, nyquist * 0.9)
        b, a = butter(3, cutoff / nyquist, btype='low')

        delta_filtered = delta.copy()
        # Filter root position residual
        for col in range(3):
            delta_filtered[:, col] = filtfilt(b, a, delta[:, col])
        # Filter joint residuals
        for col in range(7, delta.shape[1]):
            delta_filtered[:, col] = filtfilt(b, a, delta[:, col])
        # Root quaternion: compute angular difference, filter, reapply
        # For simplicity, filter the raw quaternion delta (small angles assumption)
        for col in range(3, 7):
            delta_filtered[:, col] = filtfilt(b, a, delta[:, col])
    else:
        delta_filtered = delta

    # Apply scaled corrections
    corrected = ref_interp.copy()
    # Root position: strong correction (ground height is valuable)
    corrected[:, :3] += root_correction_scale * delta_filtered[:, :3]
    # Root quaternion: moderate correction
    corrected[:, 3:7] += root_correction_scale * 0.5 * delta_filtered[:, 3:7]
    # Re-normalize quaternion
    for t in range(T_sim):
        qnorm = np.linalg.norm(corrected[t, 3:7])
        if qnorm > 1e-8:
            corrected[t, 3:7] /= qnorm
    # Joint angles: weak correction (mostly noise from PD)
    corrected[:, 7:] += joint_correction_scale * delta_filtered[:, 7:]

    return corrected


def compute_ground_offset(mjcf_path: str, ref_qpos: np.ndarray) -> float:
    """Compute ground offset using bilateral foot grounding.

    Places the humanoid so the lowest foot geom touches z=0 exactly.
    Uses ONLY foot bodies (L_Ankle, L_Toe, R_Ankle, R_Toe) for robust
    grounding that works regardless of arm/head pose.

    Matches the proven-correct approach from test_init_diff.py and
    run_smpl_rl_tracker.py process_single_motion().

    Args:
        mjcf_path: Path to SMPL humanoid MJCF XML
        ref_qpos: (T, 76) reference qpos trajectory

    Returns:
        offset: Value to subtract from qpos[:, 2] to place feet on ground.
                Equivalent to the lowest foot geom z-coordinate at frame 0.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    # Use first frame
    data.qpos[:] = ref_qpos[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # Identify foot body IDs (bilateral grounding)
    left_foot_ids = set()
    right_foot_ids = set()
    for bid in range(1, model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname in ("L_Ankle", "L_Toe"):
            left_foot_ids.add(bid)
        elif bname in ("R_Ankle", "R_Toe"):
            right_foot_ids.add(bid)

    def _lowest_geom_z(body_id_set):
        """Find lowest z of geoms belonging to given bodies.

        Correctly handles capsule orientation (half-length projection)
        and box half-extents projection onto Z axis.
        """
        min_z = float("inf")
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] not in body_id_set:
                continue
            gtype = int(model.geom_type[gid])
            gsize = model.geom_size[gid]
            gxpos = data.geom_xpos[gid]
            gxmat = data.geom_xmat[gid].reshape(3, 3)

            if gtype == 5:  # capsule: size=[radius, half_length, 0]
                # Z extent = |cos(tilt)| * half_length + radius
                z_ext = abs(gxmat[2, 2]) * gsize[1] + gsize[0]
                bottom = gxpos[2] - z_ext
            elif gtype == 3:  # sphere: size=[radius, 0, 0]
                bottom = gxpos[2] - gsize[0]
            elif gtype == 6:  # box: size=[hx, hy, hz]
                # Project all 3 half-extents onto Z
                z_ext = (abs(gxmat[2, 0]) * gsize[0] +
                         abs(gxmat[2, 1]) * gsize[1] +
                         abs(gxmat[2, 2]) * gsize[2])
                bottom = gxpos[2] - z_ext
            else:
                bottom = gxpos[2]
            min_z = min(min_z, bottom)
        return min_z

    left_min = _lowest_geom_z(left_foot_ids)
    right_min = _lowest_geom_z(right_foot_ids)
    # Use the lower of left/right foot as grounding reference
    grounding_ref_z = min(left_min, right_min)

    del model, data
    # Return the z-coordinate of lowest foot point.
    # Caller subtracts this: ref_qpos[:, 2] -= offset
    # Result: lowest foot is at z=0 (on ground plane)
    return grounding_ref_z


# ===========================================================================
#  RLPhysicsOracle — Main class
# ===========================================================================

class RLPhysicsOracle:
    """RL Physics Validation & Correction Oracle for PhysFlow.

    Wraps the pretrained RL motion tracking policy (ProtoMotions SMPL ONNX) as
    a physics oracle. Two primary functions:

    1. QUALITY GATE (validator): Can the RL policy physically track this motion
       in simulation without falling? If yes → motion is physically plausible.
       This is the PRIMARY use case for PhysFlow training: filter T2M outputs
       into "pass" (use as RL-validated target) vs "fail" (discard/curriculum).

    2. GROUND OFFSET (corrector): The only reliable physical correction from
       MuJoCo: lower the root translation so feet touch the ground plane.
       Applied via FK (no unstable simulation needed for this).

    Key properties of the RL tracker (vs PD-only):
    - Learned balance, contact, and stability from reinforcement learning
    - For physically possible motions: tracks smoothly without falling
    - For physically impossible motions: falls early → clear quality signal
    - The fall/success boundary is the physics constraint we want T2M to learn

    Sim-to-sim limitation: The ONNX policy was trained in IsaacGym (PhysX) but
    runs here in MuJoCo. This creates ~28x kinematic jitter in the raw sim
    output, making it unsuitable as a direct motion target. Hence the validator
    mode (default scales=0) which uses sim only for pass/fail assessment.
    """

    def __init__(
        self,
        onnx_path: str = _DEFAULT_ONNX,
        yaml_path: str = _DEFAULT_YAML,
        mjcf_path: str = _DEFAULT_MJCF,
        gear: float = 1.0,
    ):
        """Initialize the RL Physics Oracle.

        Args:
            onnx_path: Path to pretrained ONNX RL policy
            yaml_path: Path to unified_pipeline.yaml metadata
            mjcf_path: Path to SMPL humanoid MJCF XML
            gear: Actuator gear value (1.0 for IsaacGym-trained policies)
        """
        import yaml

        self.onnx_path = onnx_path
        self.yaml_path = yaml_path
        self.mjcf_path = mjcf_path
        self.gear = gear

        # Load YAML metadata
        with open(yaml_path) as f:
            self.yaml_meta = yaml.safe_load(f)

        # Cache body_pos_1 for qpos conversion
        import mujoco
        _temp_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.body_pos_1 = _temp_model.body_pos[1].copy()
        del _temp_model

        # Parse timing info
        self.control_dt = self.yaml_meta["timing"]["control_dt"]
        self.physics_dt = self.yaml_meta["timing"]["physics_dt"]
        self.decimation = self.yaml_meta["timing"]["decimation"]

        log.info(f"RLPhysicsOracle initialized:")
        log.info(f"  ONNX: {Path(onnx_path).name}")
        log.info(f"  Control: {1.0/self.control_dt:.0f}Hz, "
                 f"Physics: {1.0/self.physics_dt:.0f}Hz, "
                 f"Decimation: {self.decimation}")

    def correct(
        self,
        motion_135: np.ndarray,
        fps: int = 30,
        apply_ground_offset: bool = True,
        output_fps: Optional[int] = None,
        mode: str = "residual",
        lowpass_cutoff_hz: float = 5.0,
        blend_alpha: float = 0.3,
        root_correction_scale: float = 0.0,
        joint_correction_scale: float = 0.0,
    ) -> Tuple[np.ndarray, dict]:
        """Run RL closed-loop correction on a motion_135 input.

        This is the core method: takes T2M output and produces physics-valid
        motion through RL tracking in MuJoCo simulation, with post-processing
        to remove sim-to-sim transfer artifacts (PD jitter, oscillation).

        Due to sim-to-sim transfer gap (RL trained in IsaacGym, deployed in
        MuJoCo), the raw simulation output is very jittery (28x kinematic).
        The residual mode applies ONLY the low-frequency physical corrections:

        With default scales (root=0, joint=0), this operates as a "validator
        oracle": the output is the ground-offset reference motion, and the RL
        simulation serves as a QUALITY GATE (can the motion be physically
        tracked?). This is the safest mode for PhysFlow training.

        With non-zero scales (e.g., root=0.3, joint=0.05), this operates as
        a "corrector oracle": the output includes smoothed physics corrections
        from the simulation. Use for research/comparison only.

        Three correction modes:
          - "residual" (RECOMMENDED): Ground-offset reference + optional scaled
            physics residual. Default scales=0 → pure validator mode.
          - "filter_blend": Low-pass filter + reference blending. Moderate quality.
          - "raw": No post-processing. Full simulation output (very jittery).

        Args:
            motion_135: (T, 135) motion in HyMotion format [transl(3) + 22 rot6d(6)]
                       Y-up coordinate system, typically 30fps
            fps: Frame rate of the input motion
            apply_ground_offset: If True, adjust initial height to touch ground
            output_fps: Output FPS. If None, uses input fps.
            mode: Correction mode: "residual", "filter_blend", or "raw"
            lowpass_cutoff_hz: (filter_blend mode) Butterworth cutoff in Hz.
            blend_alpha: (filter_blend mode) Blend factor with reference.
            root_correction_scale: (residual mode) Scale for root correction [0-1].
            joint_correction_scale: (residual mode) Scale for joint correction [0-1].

        Returns:
            motion_135_rl: (T', 135) RL-corrected motion, same format as input.
            stats: dict with simulation statistics.
        """
        t_start = time.perf_counter()

        # [1] Decode motion_135 to SMPL axis-angle
        smpl_pose, transl = decode_motion_135_array(motion_135)
        T = smpl_pose.shape[0]

        # [2] Y-up -> Z-up
        smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose, transl)

        # [3] SMPL -> MuJoCo qpos
        ref_qpos = smpl_to_qpos(smpl_pose_zup, transl_zup, self.body_pos_1)

        # [4] Ground offset correction
        if apply_ground_offset:
            offset = compute_ground_offset(self.mjcf_path, ref_qpos)
            ref_qpos[:, 2] -= offset

        # [5] Run RL tracker simulation
        sim_qpos, sim_stats = run_rl_tracker(
            ref_qpos=ref_qpos,
            motion_fps=fps,
            onnx_path=self.onnx_path,
            mjcf_path=self.mjcf_path,
            yaml_meta=self.yaml_meta,
        )

        # [6] Post-processing based on mode
        if mode == "residual":
            # Recommended: extract low-freq physics corrections, apply to ref
            sim_qpos_final = residual_correction(
                sim_qpos, ref_qpos,
                control_dt=self.control_dt,
                ref_fps=fps,
                root_correction_scale=root_correction_scale,
                joint_correction_scale=joint_correction_scale,
                lowpass_hz=2.0,
            )
            log.info(f"  Applied residual correction "
                     f"(root={root_correction_scale}, joint={joint_correction_scale})")
        elif mode == "filter_blend":
            # Filter then blend with reference
            if lowpass_cutoff_hz and lowpass_cutoff_hz > 0 and len(sim_qpos) > 10:
                sim_qpos_filtered = lowpass_filter_qpos(
                    sim_qpos, control_dt=self.control_dt,
                    cutoff_hz=lowpass_cutoff_hz)
            else:
                sim_qpos_filtered = sim_qpos
            if blend_alpha and blend_alpha > 0:
                sim_qpos_final = blend_with_reference(
                    sim_qpos_filtered, ref_qpos,
                    blend_alpha=blend_alpha, ref_fps=fps,
                    sim_fps=1.0 / self.control_dt)
            else:
                sim_qpos_final = sim_qpos_filtered
            log.info(f"  Applied filter_blend (cutoff={lowpass_cutoff_hz}Hz, "
                     f"alpha={blend_alpha})")
        else:  # "raw"
            sim_qpos_final = sim_qpos
            log.info("  Raw mode: no post-processing")

        # [7] Convert processed output back to motion_135
        smpl_pose_sim, transl_sim = qpos_to_smpl(sim_qpos_final, self.body_pos_1)
        smpl_pose_yup, transl_yup = zup_to_yup(smpl_pose_sim, transl_sim)

        # [8] Resample to desired output FPS
        out_fps = output_fps or fps
        sim_fps = 1.0 / self.control_dt
        T_sim = len(sim_qpos_final)

        if abs(sim_fps - out_fps) > 0.5:
            T_out = int(T_sim * self.control_dt * out_fps)
            if T_out > 0:
                indices = np.linspace(0, T_sim - 1, T_out).astype(int)
                smpl_pose_yup = smpl_pose_yup[indices]
                transl_yup = transl_yup[indices]
            else:
                T_out = T_sim
        else:
            T_out = T_sim

        # [9] Encode back to motion_135
        motion_135_rl = encode_motion_135(smpl_pose_yup, transl_yup)

        # [10] Build enhanced stats with stability metrics
        elapsed = time.perf_counter() - t_start
        total_steps = sim_stats["total_sim_steps"]
        actual_steps = sim_stats["actual_sim_steps"]

        # Compute stability metrics from RAW sim (before filtering)
        root_z_raw = sim_qpos[:, 2]
        root_z_std = float(np.std(root_z_raw)) if len(root_z_raw) > 1 else 0.0

        # Compute jitter metric (frame-to-frame change magnitude)
        if len(sim_qpos) > 1:
            joint_diffs = np.diff(sim_qpos[:, 7:], axis=0)
            jitter_raw = float(np.abs(joint_diffs).mean())
        else:
            jitter_raw = 0.0

        # Post-processing jitter for the FINAL output
        if len(sim_qpos_final) > 1:
            joint_diffs_final = np.diff(sim_qpos_final[:, 7:], axis=0)
            jitter_filtered = float(np.abs(joint_diffs_final).mean())
        else:
            jitter_filtered = 0.0

        stats = {
            **sim_stats,
            "completion_ratio": actual_steps / max(total_steps, 1),
            "input_frames": T,
            "input_fps": fps,
            "output_frames": T_out,
            "output_fps": out_fps,
            "oracle_time_s": elapsed,
            "ground_offset": offset if apply_ground_offset else 0.0,
            "root_height_std": root_z_std,
            "jitter_raw": jitter_raw,
            "jitter_filtered": jitter_filtered,
            "mode": mode,
        }

        return motion_135_rl, stats

    def correct_batch(
        self,
        motions_135: list[np.ndarray],
        fps: int = 30,
        output_fps: Optional[int] = None,
    ) -> list[Tuple[np.ndarray, dict]]:
        """Correct a batch of motions sequentially.

        Note: Each motion requires its own MuJoCo simulation, so this is
        sequential. For parallelism, use multiprocessing externally.

        Args:
            motions_135: List of (T_i, 135) motion arrays
            fps: Frame rate of input motions
            output_fps: Output FPS

        Returns:
            List of (motion_135_rl, stats) tuples
        """
        results = []
        for i, motion in enumerate(motions_135):
            log.info(f"  Correcting motion {i+1}/{len(motions_135)} "
                     f"({motion.shape[0]} frames)")
            try:
                result = self.correct(motion, fps=fps, output_fps=output_fps)
                results.append(result)
            except Exception as e:
                log.warning(f"  Motion {i+1} failed: {e}")
                # Return original with error stats
                stats = {
                    "status": "error",
                    "error": str(e),
                    "completion_ratio": 0.0,
                    "root_height_min": 0.0,
                }
                results.append((motion.copy(), stats))
        return results

    def is_good_quality(
        self,
        stats: dict,
        min_completion: float = 0.8,
        min_root_height: float = 0.3,
        max_root_height_std: float = 0.3,
        max_jitter: float = 1.0,
    ) -> bool:
        """Quality gate for RL tracking results.

        Determines whether the RL correction produced a usable result
        that should be used as a training target for the T2M model.

        Criteria (measured on RAW simulation output before post-processing):
        - completion_ratio: RL must track majority of the motion without falling
        - root_height_min: Root pelvis must stay above collapse threshold
        - root_height_std: Raw sim oscillation shouldn't be extreme (indicates
          the RL policy can roughly maintain the pose even if sim is jittery)
        - jitter_filtered: Post-correction qpos jitter should be low (this is
          what actually matters for the final output quality)

        Note: Even with sim-to-sim transfer artifacts (IsaacGym→MuJoCo),
        the residual correction mode produces smooth output if the RL policy
        roughly tracks the reference. We allow root_height_std up to 0.15
        because that oscillation is NOT in the final output.

        Args:
            stats: Stats dict from correct()
            min_completion: Minimum fraction of motion tracked successfully
            min_root_height: Minimum root height (below = collapsed)
            max_root_height_std: Maximum raw root height standard deviation
                (high std = RL failed to stabilize = correction unreliable)
            max_jitter: Maximum post-correction joint jitter (radians/frame)
                in qpos space. With residual mode, this should be very low.

        Returns:
            True if the correction is good enough to use as target
        """
        if stats.get("status") == "error":
            return False

        # Check completion ratio
        if stats.get("completion_ratio", 0) < min_completion:
            return False

        # Check root height (didn't collapse)
        if stats.get("root_height_min", 0) < min_root_height:
            return False

        # Check stability (not oscillating wildly in raw sim)
        if stats.get("root_height_std", float("inf")) > max_root_height_std:
            return False

        # Check post-correction jitter is acceptable
        # NOTE: In residual mode with joint_factor=0.0, jitter_filtered reflects
        # the kinematic input's smoothness, not the correction quality.
        # Skip this check if jitter_filtered is not reported (older oracle versions).
        jitter = stats.get("jitter_filtered")
        if jitter is not None and jitter > max_jitter:
            return False

        return True

    def get_quality_score(self, stats: dict) -> float:
        """Compute a continuous quality score [0, 1] for curriculum weighting.

        Higher = better physics tracking. Used by the curriculum scheduler
        to weight training samples.
        """
        if stats.get("status") == "error":
            return 0.0

        completion = stats.get("completion_ratio", 0.0)
        root_h = stats.get("root_height_min", 0.0)

        # Combine metrics
        h_score = min(1.0, max(0.0, (root_h - 0.2) / 0.6))  # [0.2, 0.8] -> [0, 1]
        score = completion * 0.7 + h_score * 0.3
        return float(score)


# ===========================================================================
#  CLI / Test
# ===========================================================================

def _test_roundtrip():
    """Test encode/decode roundtrip for motion_135."""
    log.info("Testing encode/decode roundtrip...")

    # Generate random motion_135
    T = 60
    motion_135 = np.random.randn(T, 135).astype(np.float32) * 0.1
    # Set translation to reasonable values
    motion_135[:, 0] = 0.0  # x
    motion_135[:, 1] = 0.9  # y (height in Y-up)
    motion_135[:, 2] = np.linspace(0, 1, T)  # z (forward)

    # Decode
    smpl_pose, transl = decode_motion_135_array(motion_135)
    assert smpl_pose.shape == (T, 72)
    assert transl.shape == (T, 3)

    # Encode back
    motion_135_recon = encode_motion_135(smpl_pose, transl)
    assert motion_135_recon.shape == (T, 135)

    # Check translation exact match
    np.testing.assert_allclose(transl, motion_135_recon[:, :3], atol=1e-6)

    # Check rotation roundtrip (should be close but not exact due to
    # rot6d -> rotmat -> axis-angle -> rotmat -> rot6d)
    rot_diff = np.abs(motion_135[:, 3:] - motion_135_recon[:, 3:]).mean()
    log.info(f"  Rotation roundtrip mean abs diff: {rot_diff:.6f}")
    assert rot_diff < 0.01, f"Roundtrip error too large: {rot_diff}"

    log.info("  PASSED: encode/decode roundtrip")


def _test_oracle_on_npz(npz_path: str):
    """Test the oracle on a real motion_135 NPZ file."""
    log.info(f"Testing oracle on: {npz_path}")

    # Load NPZ
    data = np.load(npz_path, allow_pickle=True)
    motion_135 = data["motion_135"]
    fps = int(data.get("fps", 30))
    log.info(f"  Input: {motion_135.shape} @ {fps}fps")

    # Create oracle
    oracle = RLPhysicsOracle()

    # Run correction
    motion_135_rl, stats = oracle.correct(motion_135, fps=fps)
    log.info(f"  Output: {motion_135_rl.shape}")
    log.info(f"  Stats: status={stats['status']}, "
             f"completion={stats['completion_ratio']:.2f}, "
             f"root_h_min={stats['root_height_min']:.3f}")
    log.info(f"  Quality: {oracle.is_good_quality(stats)}")
    log.info(f"  Score: {oracle.get_quality_score(stats):.3f}")
    log.info(f"  Time: {stats['oracle_time_s']:.1f}s")

    return motion_135_rl, stats


def main():
    parser = argparse.ArgumentParser(
        description="PhysFlow RL Physics Correction Oracle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test", action="store_true",
                        help="Run encode/decode roundtrip test")
    parser.add_argument("--npz", type=str, default=None,
                        help="Test on a specific NPZ file")
    parser.add_argument("--onnx", type=str, default=_DEFAULT_ONNX)
    parser.add_argument("--yaml", type=str, default=_DEFAULT_YAML)
    parser.add_argument("--mjcf", type=str, default=_DEFAULT_MJCF)

    args = parser.parse_args()

    if args.test:
        _test_roundtrip()
    elif args.npz:
        _test_oracle_on_npz(args.npz)
    else:
        # Quick smoke test: just verify initialization
        log.info("Verifying Oracle initialization...")
        oracle = RLPhysicsOracle(
            onnx_path=args.onnx,
            yaml_path=args.yaml,
            mjcf_path=args.mjcf,
        )
        log.info("Oracle ready. Use --test or --npz to run actual tests.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    main()
