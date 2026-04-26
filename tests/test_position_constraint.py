"""Tests for position constraint system: FK, IK solvers, and constraint solver.

Run with:
    python -m pytest tests/test_position_constraint.py -v
    python -m pytest tests/test_position_constraint.py -v -k "test_fk"
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

# ========================================================================
# Fixtures
# ========================================================================

# SMPL-22 parents for reference
SMPL22_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19
]

NUM_JOINTS = 22


def _make_synthetic_bone_offsets() -> Tensor:
    """Create synthetic but reasonable bone offsets for SMPL-22.

    Based on approximate human skeleton proportions.
    """
    offsets = torch.zeros(22, 3)
    # Root at origin (close to pelvis height)
    offsets[0] = torch.tensor([0.0, 0.0, 0.95])

    # Pelvis children
    offsets[1] = torch.tensor([0.09, 0.0, -0.08])    # L_Hip
    offsets[2] = torch.tensor([-0.09, 0.0, -0.08])   # R_Hip
    offsets[3] = torch.tensor([0.0, 0.0, 0.13])      # Spine1

    # Leg joints
    offsets[4] = torch.tensor([0.0, 0.0, -0.42])     # L_Knee
    offsets[5] = torch.tensor([0.0, 0.0, -0.42])     # R_Knee
    offsets[6] = torch.tensor([0.0, 0.0, 0.15])      # Spine2

    offsets[7] = torch.tensor([0.0, 0.0, -0.40])     # L_Ankle
    offsets[8] = torch.tensor([0.0, 0.0, -0.40])     # R_Ankle
    offsets[9] = torch.tensor([0.0, 0.0, 0.12])      # Spine3

    offsets[10] = torch.tensor([0.0, 0.12, -0.05])   # L_Foot
    offsets[11] = torch.tensor([0.0, 0.12, -0.05])   # R_Foot
    offsets[12] = torch.tensor([0.0, 0.0, 0.12])     # Neck

    offsets[13] = torch.tensor([0.06, 0.0, 0.02])    # L_Collar
    offsets[14] = torch.tensor([-0.06, 0.0, 0.02])   # R_Collar
    offsets[15] = torch.tensor([0.0, 0.0, 0.12])     # Head

    offsets[16] = torch.tensor([0.15, 0.0, 0.0])     # L_Shoulder
    offsets[17] = torch.tensor([-0.15, 0.0, 0.0])    # R_Shoulder
    offsets[18] = torch.tensor([0.28, 0.0, 0.0])     # L_Elbow
    offsets[19] = torch.tensor([-0.28, 0.0, 0.0])    # R_Elbow
    offsets[20] = torch.tensor([0.25, 0.0, 0.0])     # L_Wrist
    offsets[21] = torch.tensor([-0.25, 0.0, 0.0])    # R_Wrist

    return offsets


def _make_identity_rotmat() -> Tensor:
    """Create identity local rotation matrices for all joints."""
    return torch.eye(3).unsqueeze(0).expand(22, -1, -1).clone()


def _make_random_motion(T: int = 10, bone_offsets: Tensor = None) -> Tensor:
    """Create random but valid 135-dim motion."""
    from hftrainer.pipelines.motion.differentiable_fk import fk_to_motion135

    motion = torch.zeros(T, 135)
    for t in range(T):
        local_rotmat = _make_identity_rotmat()
        # Add small random rotations
        for j in range(22):
            angle = torch.randn(3) * 0.1
            from hftrainer.models.motion.hymotion_m2m.network.geometry import axis_angle_to_matrix
            local_rotmat[j] = axis_angle_to_matrix(angle)

        translation = torch.tensor([0.0, float(t) * 0.05, 0.0])
        motion[t] = fk_to_motion135(local_rotmat, translation)

    return motion


# ========================================================================
# FK Tests
# ========================================================================

class TestDifferentiableFK:
    """Tests for differentiable forward kinematics."""

    def test_identity_pose(self):
        """FK with identity rotations should place joints at T-pose positions."""
        from hftrainer.pipelines.motion.differentiable_fk import differentiable_fk

        bone_offsets = _make_synthetic_bone_offsets()
        local_rotmat = _make_identity_rotmat()
        translation = torch.zeros(3)

        world_pos, world_rot = differentiable_fk(local_rotmat, translation, bone_offsets)

        assert world_pos.shape == (22, 3)
        assert world_rot.shape == (22, 3, 3)

        # Root should be at bone_offsets[0]
        assert torch.allclose(world_pos[0], bone_offsets[0], atol=1e-6)

        # All world rotations should be identity
        eye = torch.eye(3)
        for j in range(22):
            assert torch.allclose(world_rot[j], eye, atol=1e-6), f"Joint {j} not identity"

    def test_translation_offset(self):
        """Translation should shift all joints equally."""
        from hftrainer.pipelines.motion.differentiable_fk import differentiable_fk

        bone_offsets = _make_synthetic_bone_offsets()
        local_rotmat = _make_identity_rotmat()
        trans = torch.tensor([1.0, 2.0, 3.0])

        world_pos_zero, _ = differentiable_fk(local_rotmat, torch.zeros(3), bone_offsets)
        world_pos_trans, _ = differentiable_fk(local_rotmat, trans, bone_offsets)

        for j in range(22):
            diff = world_pos_trans[j] - world_pos_zero[j]
            assert torch.allclose(diff, trans, atol=1e-6)

    def test_batch_shape(self):
        """FK should handle batched inputs."""
        from hftrainer.pipelines.motion.differentiable_fk import differentiable_fk

        bone_offsets = _make_synthetic_bone_offsets()
        B, T = 2, 5
        local_rotmat = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, 22, -1, -1).clone()
        translation = torch.zeros(B, T, 3)

        world_pos, world_rot = differentiable_fk(local_rotmat, translation, bone_offsets)

        assert world_pos.shape == (B, T, 22, 3)
        assert world_rot.shape == (B, T, 22, 3, 3)

    def test_fk_gradient(self):
        """FK should be differentiable."""
        from hftrainer.pipelines.motion.differentiable_fk import differentiable_fk

        bone_offsets = _make_synthetic_bone_offsets()
        local_rotmat = _make_identity_rotmat().requires_grad_(True)
        translation = torch.zeros(3, requires_grad=True)

        world_pos, _ = differentiable_fk(local_rotmat, translation, bone_offsets)
        loss = world_pos[20].sum()  # L_Wrist position
        loss.backward()

        assert local_rotmat.grad is not None
        assert translation.grad is not None

    def test_motion135_roundtrip(self):
        """motion135 -> FK -> motion135 roundtrip should be lossless."""
        from hftrainer.pipelines.motion.differentiable_fk import (
            fk_to_motion135,
            motion135_to_fk,
        )

        bone_offsets = _make_synthetic_bone_offsets()
        local_rotmat = _make_identity_rotmat()
        translation = torch.tensor([0.5, 1.0, 0.0])
        motion_orig = fk_to_motion135(local_rotmat, translation)

        # Round-trip
        _, _, trans_out, rotmat_out = motion135_to_fk(motion_orig, bone_offsets)
        motion_back = fk_to_motion135(rotmat_out, trans_out)

        assert torch.allclose(motion_orig, motion_back, atol=1e-5), (
            f"Roundtrip error: {(motion_orig - motion_back).abs().max().item()}"
        )


# ========================================================================
# IK Tests
# ========================================================================

class TestRootIK:
    """Tests for root (Pelvis) position constraint."""

    def test_root_constraint(self):
        """Root constraint should be exact (just translation adjustment)."""
        from hftrainer.pipelines.motion.differentiable_fk import differentiable_fk
        from hftrainer.pipelines.motion.ik_solver import solve_root_ik

        bone_offsets = _make_synthetic_bone_offsets()
        local_rotmat = _make_identity_rotmat()
        translation = torch.zeros(3)
        target = torch.tensor([1.0, 2.0, 3.0])

        new_trans = solve_root_ik(translation, bone_offsets, target)
        world_pos, _ = differentiable_fk(local_rotmat, new_trans, bone_offsets)

        error = (world_pos[0] - target).norm().item()
        assert error < 1e-5, f"Root error: {error:.6f}m (expected < 0.01mm)"


class TestTwoBoneIK:
    """Tests for 2-bone analytic IK (wrist, ankle, foot)."""

    @pytest.mark.parametrize("target_joint", [20, 21, 7, 8, 10, 11])
    def test_two_bone_reachable(self, target_joint):
        """2-bone IK should reach targets within bone length limits."""
        from hftrainer.pipelines.motion.differentiable_fk import differentiable_fk
        from hftrainer.pipelines.motion.ik_solver import solve_two_bone_ik, TWO_BONE_CHAINS

        bone_offsets = _make_synthetic_bone_offsets()
        local_rotmat = _make_identity_rotmat()
        translation = torch.zeros(3)
        world_pos, _ = differentiable_fk(local_rotmat, translation, bone_offsets)

        base, mid, end = TWO_BONE_CHAINS[target_joint]
        base_pos = world_pos[base]
        arm_length = bone_offsets[mid].norm() + bone_offsets[end].norm()

        # Target: 80% of arm length from base, slightly offset from current direction
        current_dir = F.normalize(world_pos[end] - base_pos, dim=-1)
        # Build a perpendicular direction
        perp = torch.tensor([0.0, 1.0, 0.0])
        if (perp - current_dir).norm() < 0.1:
            perp = torch.tensor([1.0, 0.0, 0.0])
        perp = F.normalize(perp - (perp * current_dir).sum() * current_dir, dim=-1)

        target = base_pos + current_dir * arm_length * 0.7 + perp * arm_length * 0.3

        new_rotmat = solve_two_bone_ik(
            local_rotmat, translation, bone_offsets, target_joint, target
        )
        new_world_pos, _ = differentiable_fk(new_rotmat, translation, bone_offsets)

        error = (new_world_pos[target_joint] - target).norm().item()
        assert error < 0.005, f"2-bone IK error for joint {target_joint}: {error*1000:.2f}mm"

    def test_two_bone_unreachable(self):
        """2-bone IK should not crash on unreachable targets."""
        from hftrainer.pipelines.motion.differentiable_fk import differentiable_fk
        from hftrainer.pipelines.motion.ik_solver import solve_two_bone_ik

        bone_offsets = _make_synthetic_bone_offsets()
        local_rotmat = _make_identity_rotmat()
        translation = torch.zeros(3)

        # Target way too far (10 meters away)
        target = torch.tensor([10.0, 0.0, 0.0])

        # Should not crash
        new_rotmat = solve_two_bone_ik(
            local_rotmat, translation, bone_offsets, 20, target
        )
        assert new_rotmat.shape == (22, 3, 3)
        assert not torch.isnan(new_rotmat).any()


class TestOneBoneIK:
    """Tests for 1-bone IK (hip, knee, spine1)."""

    @pytest.mark.parametrize("target_joint", [1, 2, 3, 4, 5])
    def test_one_bone(self, target_joint):
        """1-bone IK should get close to target."""
        from hftrainer.pipelines.motion.differentiable_fk import differentiable_fk
        from hftrainer.pipelines.motion.ik_solver import solve_one_bone_ik

        bone_offsets = _make_synthetic_bone_offsets()
        local_rotmat = _make_identity_rotmat()
        translation = torch.zeros(3)

        world_pos, _ = differentiable_fk(local_rotmat, translation, bone_offsets)
        current_pos = world_pos[target_joint]

        # Move target slightly (within bone length)
        target = current_pos + torch.tensor([0.02, 0.02, 0.0])

        new_rotmat = solve_one_bone_ik(
            local_rotmat, translation, bone_offsets, target_joint, target
        )
        new_world_pos, _ = differentiable_fk(new_rotmat, translation, bone_offsets)

        error = (new_world_pos[target_joint] - target).norm().item()
        # 1-bone can only do direction, distance is bone-length dependent
        # Check direction is correct
        assert error < 0.05, f"1-bone IK error for joint {target_joint}: {error:.4f}m"


class TestGradientIK:
    """Tests for gradient-based IK (spine, head, collar, etc)."""

    @pytest.mark.parametrize("target_joint", [12, 15, 6, 9])
    def test_gradient_ik(self, target_joint):
        """Gradient IK should converge for spine/neck/head."""
        from hftrainer.pipelines.motion.differentiable_fk import differentiable_fk
        from hftrainer.pipelines.motion.ik_solver import solve_gradient_ik

        bone_offsets = _make_synthetic_bone_offsets()
        local_rotmat = _make_identity_rotmat()
        translation = torch.zeros(3)

        world_pos, _ = differentiable_fk(local_rotmat, translation, bone_offsets)
        current_pos = world_pos[target_joint]

        # Small displacement target
        target = current_pos + torch.tensor([0.02, 0.01, 0.0])

        new_rotmat = solve_gradient_ik(
            local_rotmat, translation, bone_offsets,
            target_joint, target,
            lr=0.02, num_steps=100, tol=1e-4,
        )
        new_world_pos, _ = differentiable_fk(new_rotmat, translation, bone_offsets)

        error = (new_world_pos[target_joint] - target).norm().item()
        assert error < 0.002, f"Gradient IK error for joint {target_joint}: {error*1000:.2f}mm"


# ========================================================================
# Position Constraint Solver Tests
# ========================================================================

class TestPositionConstraintSolver:
    """Tests for the unified constraint solver."""

    def test_root_constraint(self):
        """Root position constraint should have sub-mm error."""
        from hftrainer.pipelines.motion.position_constraint import (
            PositionConstraint,
            PositionConstraintSolver,
        )

        bone_offsets = _make_synthetic_bone_offsets()
        solver = PositionConstraintSolver(bone_offsets)
        motion = _make_random_motion(10, bone_offsets)

        target = torch.tensor([1.0, 2.0, 0.5])
        constraints = [PositionConstraint(frame=5, joint=0, target_xyz=target)]

        motion_fixed, max_error = solver.solve(motion, constraints)
        assert max_error < 1e-4, f"Root constraint error: {max_error*1000:.2f}mm"

    def test_wrist_constraint(self):
        """Wrist position constraint should have small error."""
        from hftrainer.pipelines.motion.position_constraint import (
            PositionConstraint,
            PositionConstraintSolver,
        )
        from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk

        bone_offsets = _make_synthetic_bone_offsets()
        solver = PositionConstraintSolver(bone_offsets)
        motion = _make_random_motion(10, bone_offsets)

        # Get current wrist position and move it slightly (small displacement)
        world_pos, _, _, _ = motion135_to_fk(motion[5], bone_offsets)
        target = world_pos[20] + torch.tensor([0.0, 0.02, 0.0])
        constraints = [PositionConstraint(frame=5, joint=20, target_xyz=target)]

        motion_fixed, max_error = solver.solve(motion, constraints)
        assert max_error < 0.01, f"Wrist constraint error: {max_error*1000:.2f}mm"

    def test_ankle_constraint(self):
        """Ankle position constraint should have small error."""
        from hftrainer.pipelines.motion.position_constraint import (
            PositionConstraint,
            PositionConstraintSolver,
        )
        from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk

        bone_offsets = _make_synthetic_bone_offsets()
        solver = PositionConstraintSolver(bone_offsets)
        motion = _make_random_motion(10, bone_offsets)

        world_pos, _, _, _ = motion135_to_fk(motion[3], bone_offsets)
        target = world_pos[7] + torch.tensor([0.0, 0.02, 0.0])
        constraints = [PositionConstraint(frame=3, joint=7, target_xyz=target)]

        motion_fixed, max_error = solver.solve(motion, constraints)
        assert max_error < 0.01, f"Ankle constraint error: {max_error*1000:.2f}mm"

    def test_neck_gradient_constraint(self):
        """Neck (gradient IK) constraint should converge."""
        from hftrainer.pipelines.motion.position_constraint import (
            PositionConstraint,
            PositionConstraintSolver,
        )
        from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk

        bone_offsets = _make_synthetic_bone_offsets()
        solver = PositionConstraintSolver(bone_offsets)
        motion = _make_random_motion(10, bone_offsets)

        world_pos, _, _, _ = motion135_to_fk(motion[5], bone_offsets)
        target = world_pos[12] + torch.tensor([0.02, 0.01, 0.0])
        constraints = [PositionConstraint(frame=5, joint=12, target_xyz=target)]

        motion_fixed, max_error = solver.solve(motion, constraints)
        assert max_error < 0.005, f"Neck constraint error: {max_error*1000:.2f}mm"

    def test_head_gradient_constraint(self):
        """Head (gradient IK) constraint should converge."""
        from hftrainer.pipelines.motion.position_constraint import (
            PositionConstraint,
            PositionConstraintSolver,
        )
        from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk

        bone_offsets = _make_synthetic_bone_offsets()
        solver = PositionConstraintSolver(bone_offsets)
        motion = _make_random_motion(10, bone_offsets)

        world_pos, _, _, _ = motion135_to_fk(motion[5], bone_offsets)
        target = world_pos[15] + torch.tensor([0.02, 0.01, 0.0])
        constraints = [PositionConstraint(frame=5, joint=15, target_xyz=target)]

        motion_fixed, max_error = solver.solve(motion, constraints)
        assert max_error < 0.01, f"Head constraint error: {max_error*1000:.2f}mm"

    def test_multi_constraint(self):
        """Multiple constraints on different chains should work together."""
        from hftrainer.pipelines.motion.position_constraint import (
            PositionConstraint,
            PositionConstraintSolver,
        )
        from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk

        bone_offsets = _make_synthetic_bone_offsets()
        solver = PositionConstraintSolver(bone_offsets)
        motion = _make_random_motion(10, bone_offsets)

        world_pos, _, _, _ = motion135_to_fk(motion[5], bone_offsets)
        # Left wrist + right ankle (different chains) - small displacements
        constraints = [
            PositionConstraint(frame=5, joint=20, target_xyz=world_pos[20] + torch.tensor([0.0, 0.02, 0.0])),
            PositionConstraint(frame=5, joint=8, target_xyz=world_pos[8] + torch.tensor([0.0, 0.02, 0.0])),
        ]

        motion_fixed, max_error = solver.solve(motion, constraints)
        assert max_error < 0.02, f"Multi-constraint error: {max_error*1000:.2f}mm"

    def test_unreachable_clamp(self):
        """Unreachable targets should not cause crashes."""
        from hftrainer.pipelines.motion.position_constraint import (
            PositionConstraint,
            PositionConstraintSolver,
        )

        bone_offsets = _make_synthetic_bone_offsets()
        solver = PositionConstraintSolver(bone_offsets)
        motion = _make_random_motion(10, bone_offsets)

        # Way too far target
        constraints = [
            PositionConstraint(frame=5, joint=20, target_xyz=torch.tensor([10.0, 0.0, 0.0])),
        ]

        motion_fixed, max_error = solver.solve(motion, constraints)
        assert not torch.isnan(motion_fixed).any(), "NaN in output"
        assert torch.isfinite(motion_fixed).all(), "Inf in output"

    def test_empty_constraints(self):
        """No constraints should return unchanged motion."""
        from hftrainer.pipelines.motion.position_constraint import (
            PositionConstraintSolver,
        )

        bone_offsets = _make_synthetic_bone_offsets()
        solver = PositionConstraintSolver(bone_offsets)
        motion = _make_random_motion(10, bone_offsets)

        motion_fixed, max_error = solver.solve(motion, [])
        assert torch.allclose(motion, motion_fixed)
        assert max_error == 0.0


class TestNormalizeRoundtrip:
    """Test normalize/denormalize roundtrip precision."""

    def test_roundtrip_100_times(self):
        """100x normalize->denormalize should accumulate minimal error."""
        # Simulate with known mean/std
        mean = torch.randn(135)
        std = torch.randn(135).abs().clamp(min=0.01)

        motion = torch.randn(10, 135)
        motion_orig = motion.clone()

        for _ in range(100):
            motion = (motion - mean) / std  # normalize
            motion = motion * std + mean    # denormalize

        error = (motion - motion_orig).abs().max().item()
        assert error < 1e-4, f"100x roundtrip error: {error}"


class TestAffectedDims:
    """Test affected dimension computation."""

    def test_root_dims(self):
        """Root constraint should affect translation dims only."""
        from hftrainer.pipelines.motion.position_constraint import (
            PositionConstraint,
            get_affected_dims,
        )
        constraints = [PositionConstraint(frame=0, joint=0, target_xyz=torch.zeros(3))]
        dims = get_affected_dims(constraints)
        assert dims == [0, 1, 2]

    def test_wrist_dims(self):
        """Wrist constraint should affect shoulder and elbow rot6d dims."""
        from hftrainer.pipelines.motion.position_constraint import (
            PositionConstraint,
            get_affected_dims,
        )
        # L_Wrist(20): 2-bone chain base=L_Shoulder(16), mid=L_Elbow(18)
        constraints = [PositionConstraint(frame=0, joint=20, target_xyz=torch.zeros(3))]
        dims = get_affected_dims(constraints)

        # L_Shoulder(16): dims 3+16*6=99 to 104
        # L_Elbow(18): dims 3+18*6=111 to 116
        expected_shoulder = list(range(99, 105))
        expected_elbow = list(range(111, 117))
        assert all(d in dims for d in expected_shoulder)
        assert all(d in dims for d in expected_elbow)


# ========================================================================
# All 22 joints coverage
# ========================================================================

class TestAll22Joints:
    """Test position constraint for every SMPL-22 joint."""

    @pytest.mark.parametrize("joint_idx", list(range(22)))
    def test_joint_constraint(self, joint_idx):
        """Every joint should be constrainable without errors."""
        from hftrainer.pipelines.motion.position_constraint import (
            PositionConstraint,
            PositionConstraintSolver,
        )
        from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk

        bone_offsets = _make_synthetic_bone_offsets()
        solver = PositionConstraintSolver(bone_offsets)
        motion = _make_random_motion(5, bone_offsets)

        # Get current position and create a nearby target
        world_pos, _, _, _ = motion135_to_fk(motion[2], bone_offsets)
        target = world_pos[joint_idx] + torch.tensor([0.02, 0.01, 0.0])
        constraints = [PositionConstraint(frame=2, joint=joint_idx, target_xyz=target)]

        motion_fixed, max_error = solver.solve(motion, constraints)

        # All joints should be solvable, but accuracy varies
        assert not torch.isnan(motion_fixed).any(), f"NaN for joint {joint_idx}"
        assert torch.isfinite(motion_fixed).all(), f"Inf for joint {joint_idx}"

        # Root and 2-bone should be very accurate, others reasonable
        from hftrainer.pipelines.motion.ik_solver import get_ik_strategy
        strategy = get_ik_strategy(joint_idx)
        if strategy == 'root':
            assert max_error < 1e-4, f"Joint {joint_idx} ({strategy}): {max_error*1000:.2f}mm"
        elif strategy == 'two_bone':
            assert max_error < 0.05, f"Joint {joint_idx} ({strategy}): {max_error*1000:.2f}mm"
        elif strategy == 'one_bone':
            assert max_error < 0.05, f"Joint {joint_idx} ({strategy}): {max_error*1000:.2f}mm"
        else:  # gradient
            assert max_error < 0.025, f"Joint {joint_idx} ({strategy}): {max_error*1000:.2f}mm"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
