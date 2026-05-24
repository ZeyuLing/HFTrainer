"""PhysFlow Physics Oracle: MuJoCo PD-tracking as motion correction oracle.

Wraps the existing physics simulation pipeline (run_smpl_physics_sim.py) into
a clean API for use in PhysFlow training. The oracle takes motion_135 (T, 135)
as input and returns physics-corrected motion_135 as output.

Key pipeline:
    motion_135 (Y-up) → decode → Y→Z-up → smpl→qpos → physics sim → qpos→smpl → Z→Y-up → encode → motion_135_phys

Usage:
    oracle = PhysicsOracle("path/to/smpl_humanoid.xml")
    motion_phys, stats = oracle.correct(motion_135_array)
    if oracle.is_good_quality(stats):
        # Use motion_phys as physics-grounded target
        ...
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as sRot

# Add parent for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.embodied.run_smpl_physics_sim import (
    rot6d_to_rotmat,
    yup_to_zup,
    zup_to_yup,
    smpl_to_qpos,
    qpos_to_smpl,
    load_mujoco_model,
    compute_ground_offset,
    run_physics_sim,
    smooth_simulated_qpos,
    smooth_smpl_poses,
)


def decode_motion_135_array(motion_135: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decode motion_135 array to SMPL pose + translation (Y-up).

    Same as decode_motion_135() but takes numpy array directly instead of NPZ path.

    Args:
        motion_135: (T, 135) = transl(3) + 22 x rot6d(6), Y-up

    Returns:
        smpl_pose: (T, 72) axis-angle in SMPL joint order, Y-up
        transl:    (T, 3) translation, Y-up
    """
    T = motion_135.shape[0]
    assert motion_135.shape[1] == 135, f"Expected 135 dims, got {motion_135.shape[1]}"

    transl = motion_135[:, :3].copy()  # (T, 3)
    rot6d = motion_135[:, 3:].reshape(T, 22, 6)  # (T, 22, 6)

    # rot6d -> rotation matrix -> axis-angle
    rotmat = rot6d_to_rotmat(rot6d)  # (T, 22, 3, 3)
    aa = sRot.from_matrix(
        rotmat.reshape(-1, 3, 3)
    ).as_rotvec().reshape(T, 22, 3)  # (T, 22, 3)

    root_orient = aa[:, 0, :]  # (T, 3)
    body_pose = aa[:, 1:22, :].reshape(T, -1)  # (T, 63)

    # Build full 72-dim SMPL pose (pad joints 22-23 with zeros)
    smpl_pose = np.zeros((T, 72), dtype=np.float32)
    smpl_pose[:, :3] = root_orient
    smpl_pose[:, 3:66] = body_pose
    # joints 22-23 (L_Hand, R_Hand) = 0

    return smpl_pose, transl.astype(np.float32)


def encode_to_motion_135(smpl_pose: np.ndarray, transl: np.ndarray) -> np.ndarray:
    """Encode SMPL pose + translation back to motion_135 format.

    Inverse of decode_motion_135_array().

    Args:
        smpl_pose: (T, 72) axis-angle, Y-up
        transl:    (T, 3) translation, Y-up

    Returns:
        motion_135: (T, 135) = transl(3) + 22 x rot6d(6), row-major
    """
    T = smpl_pose.shape[0]

    # Extract 22 joints (skip hand joints 22-23)
    root_orient = smpl_pose[:, :3]  # (T, 3)
    body_pose = smpl_pose[:, 3:66].reshape(T, 21, 3)  # (T, 21, 3)

    # Combine: 22 joints
    all_aa = np.zeros((T, 22, 3), dtype=np.float32)
    all_aa[:, 0, :] = root_orient
    all_aa[:, 1:22, :] = body_pose

    # axis-angle → rotation matrix → rot6d (row-major)
    rotmat = sRot.from_rotvec(
        all_aa.reshape(-1, 3)
    ).as_matrix().reshape(T, 22, 3, 3)  # (T, 22, 3, 3)

    rot6d = _rotmat_to_rot6d_batch(rotmat)  # (T, 22, 6) row-major

    # Assemble motion_135
    motion_135 = np.zeros((T, 135), dtype=np.float32)
    motion_135[:, :3] = transl
    motion_135[:, 3:] = rot6d.reshape(T, 132)

    return motion_135


def _rotmat_to_rot6d_batch(rotmat: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to row-major rot6d.

    Args:
        rotmat: (..., 3, 3) rotation matrices

    Returns:
        rot6d: (..., 6) row-major rot6d [R00, R01, R10, R11, R20, R21]
    """
    orig_shape = rotmat.shape[:-2]
    rotmat_flat = rotmat.reshape(-1, 3, 3)
    N = rotmat_flat.shape[0]

    # Extract first two columns → column-major [R00,R10,R20, R01,R11,R21]
    col0 = rotmat_flat[:, :, 0]  # (N, 3)
    col1 = rotmat_flat[:, :, 1]  # (N, 3)
    col_major = np.concatenate([col0, col1], axis=-1)  # (N, 6)

    # Reorder: column-major → row-major [0,3,1,4,2,5]
    row_major = col_major[:, [0, 3, 1, 4, 2, 5]]  # (N, 6)

    return row_major.reshape(*orig_shape, 6)


class PhysicsOracle:
    """MuJoCo PD-tracking physics correction oracle.

    Takes motion_135 (T, 135) as input, runs PD-tracking physics simulation
    in MuJoCo to enforce physical constraints (ground contact, no penetration,
    gravity), and returns physics-corrected motion_135.

    The oracle preserves the overall motion structure while fixing:
    - Foot sliding (ground friction)
    - Ground penetration (contact constraints)
    - Floating (gravity enforcement)
    - Unnatural jitter (PD smoothing + post-processing)
    """

    def __init__(self, xml_path: str, fps: int = 30, verbose: bool = False):
        """Initialize physics oracle.

        Args:
            xml_path: Path to SMPL MuJoCo XML model
            fps: Control frame rate (default 30, matching T2M output)
            verbose: Print detailed simulation info
        """
        self.xml_path = xml_path
        self.fps = fps
        self.verbose = verbose

        # Pre-load model to get body_pos_1
        model, _ = load_mujoco_model(xml_path)
        self.body_pos_1 = model.body_pos[1].copy()

    def correct(self, motion_135: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Run full physics correction pipeline.

        Args:
            motion_135: (T, 135) raw motion, Y-up

        Returns:
            motion_135_phys: (T', 135) physics-corrected, Y-up. T' <= T (shorter if fall).
            stats: dict with quality metrics:
                - completed: bool (no fall)
                - simulated_frames: int
                - total_frames: int
                - joint_tracking_error_rad: float
                - root_position_drift_m: float
                - min_root_height_m: float
        """
        # [1] Decode motion_135 → SMPL pose + translation
        smpl_pose, transl = decode_motion_135_array(motion_135)

        # [2] Y-up → Z-up
        smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose, transl)

        # [3] SMPL → MuJoCo qpos
        model, data = load_mujoco_model(self.xml_path)
        ref_qpos = smpl_to_qpos(smpl_pose_zup, transl_zup, self.body_pos_1, model=model)

        # [3.5] Ground offset
        ground_offset = compute_ground_offset(model, data, ref_qpos)
        if abs(ground_offset) > 0.001:
            ref_qpos[:, 2] -= ground_offset

        # [4] Physics simulation (free root — root emerges from contact forces)
        sim_qpos, stats = run_physics_sim(model, data, ref_qpos, self.fps,
                                          root_mode="free")
        stats["ground_offset_m"] = float(ground_offset)

        if self.verbose:
            T_sim = stats["simulated_frames"]
            T = stats["total_frames"]
            status = "OK" if stats["completed"] else f"FELL@{stats['fall_frame']}"
            print(f"  Physics: {T_sim}/{T} frames, {status}, "
                  f"err={stats['joint_tracking_error_rad']:.4f}rad, "
                  f"root_drift={stats['root_position_drift_m']:.3f}m")

        # [4.5] Post-simulation smoothing (full physics, no blending with ref)
        sim_qpos = smooth_simulated_qpos(sim_qpos, ref_qpos[:len(sim_qpos)], self.fps,
                                         blend_alpha=1.0)

        # [5] qpos → SMPL
        smpl_pose_sim, transl_sim = qpos_to_smpl(sim_qpos, self.body_pos_1)

        # [5.3] Smooth SMPL poses (remove Euler→AA jitter)
        smpl_pose_sim = smooth_smpl_poses(smpl_pose_sim, self.fps)

        # [5.5] Undo ground offset
        if abs(ground_offset) > 0.001:
            transl_sim[:, 2] += ground_offset

        # [6] Z-up → Y-up
        smpl_pose_yup, transl_yup = zup_to_yup(smpl_pose_sim, transl_sim)

        # [7] Encode back to motion_135
        motion_135_phys = encode_to_motion_135(smpl_pose_yup, transl_yup)

        return motion_135_phys, stats

    def is_good_quality(
        self,
        stats: dict,
        min_completion_rate: float = 0.8,
        max_tracking_error: float = 0.3,
    ) -> bool:
        """Quality gate: determine if physics correction was successful.

        Args:
            stats: dict from correct()
            min_completion_rate: Minimum fraction of frames simulated (0.8 = 80%)
            max_tracking_error: Maximum joint tracking error in radians

        Returns:
            True if physics correction is usable as training target
        """
        # Check completion
        completion_rate = stats["simulated_frames"] / max(stats["total_frames"], 1)
        if completion_rate < min_completion_rate:
            return False

        # Check tracking quality
        if stats["joint_tracking_error_rad"] > max_tracking_error:
            return False

        return True


# ===========================================================================
#  Self-test
# ===========================================================================

def _test_oracle():
    """Quick self-test with synthetic motion data."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, help="Optional NPZ file to test with")
    parser.add_argument("--xml", type=str,
                        default="ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml")
    args = parser.parse_args()

    print("=" * 60)
    print("PhysicsOracle Self-Test")
    print("=" * 60)

    oracle = PhysicsOracle(args.xml, verbose=True)

    if args.npz:
        # Use real motion
        data = np.load(args.npz, allow_pickle=True)
        motion_135 = data["motion_135"]
        print(f"Loaded: {args.npz}, shape={motion_135.shape}")
    else:
        # Generate T-pose standing motion (all zeros except root height)
        T = 60
        motion_135 = np.zeros((T, 135), dtype=np.float32)
        # Set translation Y (up in Y-up) to ~0.9m (standing)
        motion_135[:, 1] = 0.9
        # Set root rotation to identity (rot6d for identity: [1,0,0,1,0,0] row-major)
        # row-major identity = [R00,R01,R10,R11,R20,R21] = [1,0,0,1,0,0]
        motion_135[:, 3] = 1.0  # R00
        motion_135[:, 6] = 1.0  # R11 (at index 3+3=6 in motion_135)
        # Actually rot6d starts at index 3, each joint is 6 dims
        # For identity: col-major=[1,0,0,0,1,0], row-major=[1,0,0,1,0,0]
        for j in range(22):
            base = 3 + j * 6
            motion_135[:, base + 0] = 1.0  # R00
            motion_135[:, base + 3] = 1.0  # R11
        print(f"Generated synthetic T-pose: shape={motion_135.shape}")

    # Run correction
    motion_phys, stats = oracle.correct(motion_135)
    print(f"\nResult:")
    print(f"  Input shape:  {motion_135.shape}")
    print(f"  Output shape: {motion_phys.shape}")
    print(f"  Completed:    {stats['completed']}")
    print(f"  Frames:       {stats['simulated_frames']}/{stats['total_frames']}")
    print(f"  Tracking err: {stats['joint_tracking_error_rad']:.4f} rad")
    print(f"  Quality OK:   {oracle.is_good_quality(stats)}")

    # Verify roundtrip (encode/decode consistency)
    smpl_pose, transl = decode_motion_135_array(motion_phys)
    motion_rt = encode_to_motion_135(smpl_pose, transl)
    max_diff = np.max(np.abs(motion_phys - motion_rt))
    print(f"  Roundtrip error: {max_diff:.6f}")
    assert max_diff < 0.01, f"Roundtrip error too large: {max_diff}"

    print("\n[PASS] PhysicsOracle self-test passed!")
    return oracle, motion_phys, stats


if __name__ == "__main__":
    _test_oracle()
