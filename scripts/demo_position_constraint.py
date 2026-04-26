"""End-to-end demo for position constraint in HyMotion M2M.

Demonstrates applying world-space position constraints to a motion sequence,
either standalone (pure IK on existing motion) or integrated with the M2M
pipeline (IK during ODE inference).

Usage::

    # Pure IK demo (no model needed):
    python scripts/demo_position_constraint.py \\
        --input_npz data/test_motion.npz \\
        --constraints "frame=30,joint=20,xyz=1.0,1.5,0.3" \\
        --output_npz output/constrained.npz

    # Pipeline demo (requires model checkpoint):
    python scripts/demo_position_constraint.py \\
        --checkpoint work_dirs/hymotion_m2m_completion_uncond_fm_man_046b/latest.ckpt \\
        --input_npz data/test_motion.npz \\
        --constraints "frame=30,joint=20,xyz=1.0,1.5,0.3" \\
        --output_npz output/constrained.npz \\
        --mode pipeline
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch


def parse_constraints(constraint_strs: list) -> list:
    """Parse constraint strings into PositionConstraint objects.

    Format: "frame=30,joint=20,xyz=1.0,1.5,0.3"
    """
    from hftrainer.pipelines.motion.position_constraint import PositionConstraint

    constraints = []
    for s in constraint_strs:
        parts = {}
        for kv in s.split(','):
            if '=' in kv:
                key, val = kv.split('=', 1)
                parts[key.strip()] = val.strip()
            else:
                # Handle xyz values that follow the xyz= key
                if 'xyz' in parts:
                    parts['xyz'] += ',' + kv.strip()

        frame = int(parts['frame'])
        joint = int(parts['joint'])
        xyz = [float(x) for x in parts['xyz'].split(',')]
        assert len(xyz) == 3, f"Expected 3 xyz values, got {len(xyz)}: {xyz}"

        constraints.append(PositionConstraint(
            frame=frame,
            joint=joint,
            target_xyz=torch.tensor(xyz, dtype=torch.float32),
        ))

    return constraints


def load_motion_from_npz(npz_path: str) -> torch.Tensor:
    """Load motion from NPZ file and convert to 135-dim format.

    Expects NPZ with 'poses' (T, 156) in axis-angle and 'trans' (T, 3).
    Returns (T, 135) denormalized motion in row-major rot6d.
    """
    data = np.load(npz_path, allow_pickle=True)

    if 'motion_135' in data:
        # Already in 135-dim format
        return torch.from_numpy(data['motion_135']).float()

    # Convert from SMPL format
    poses = data['poses']  # (T, 156) or (T, J*3)
    trans = data['trans']   # (T, 3)

    T = poses.shape[0]

    # Take first 22 joints (body only, no hands/face)
    if poses.shape[1] >= 66:
        body_aa = poses[:, :66].reshape(T, 22, 3)  # axis-angle
    else:
        raise ValueError(f"Expected at least 66 pose dims, got {poses.shape[1]}")

    from hftrainer.models.motion.hymotion_m2m.network.geometry import (
        angle_axis_to_rotation_matrix,
        rotation_matrix_to_rot6d,
    )

    body_aa_t = torch.from_numpy(body_aa).float()
    rotmat = angle_axis_to_rotation_matrix(body_aa_t)  # (T, 22, 3, 3)
    rot6d = rotation_matrix_to_rot6d(rotmat)  # (T, 22, 6) row-major

    trans_t = torch.from_numpy(trans).float()
    motion = torch.cat([trans_t, rot6d.reshape(T, 132)], dim=-1)  # (T, 135)

    return motion


def save_motion_to_npz(motion_135: torch.Tensor, output_path: str, fps: int = 30):
    """Save 135-dim motion to NPZ file."""
    from hftrainer.models.motion.hymotion_m2m.network.smpl_lite import (
        construct_smpl_data_dict,
    )

    T = motion_135.shape[0]
    trans = motion_135[:, :3]
    rot6d = motion_135[:, 3:135].reshape(T, 22, 6)

    smpl_data = construct_smpl_data_dict(rot6d, trans)
    smpl_data['mocap_framerate'] = fps

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez(output_path, **smpl_data)
    print(f"Saved constrained motion to {output_path}")


def demo_pure_ik(args):
    """Demo: apply position constraints using pure IK (no model)."""
    from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk
    from hftrainer.pipelines.motion.position_constraint import (
        PositionConstraintSolver,
    )

    # Load motion
    motion = load_motion_from_npz(args.input_npz)
    print(f"Loaded motion: {motion.shape} (T={motion.shape[0]})")

    # Load or compute bone offsets
    if os.path.isfile('data/hymotion_m2m_data/bone_offsets_22.pt'):
        bone_offsets = torch.load('data/hymotion_m2m_data/bone_offsets_22.pt')
        print(f"Loaded bone offsets from data/hymotion_m2m_data/bone_offsets_22.pt")
    else:
        from tools.precompute_bone_offsets import compute_bone_offsets_from_smpl
        bone_offsets = compute_bone_offsets_from_smpl()
        print(f"Computed bone offsets from SMPL model")

    # Parse constraints
    constraints = parse_constraints(args.constraints)
    print(f"Constraints ({len(constraints)}):")
    for c in constraints:
        print(f"  frame={c.frame}, joint={c.joint}, target={c.target_xyz.tolist()}")

    # Show pre-IK positions
    for c in constraints:
        if c.frame < motion.shape[0]:
            world_pos, _, _, _ = motion135_to_fk(motion[c.frame], bone_offsets)
            current = world_pos[c.joint]
            err = (current - c.target_xyz).norm().item()
            print(f"  Pre-IK: joint {c.joint} at frame {c.frame}: "
                  f"current={current.tolist()}, error={err*1000:.2f}mm")

    # Solve
    solver = PositionConstraintSolver(bone_offsets)
    motion_fixed, max_error = solver.solve(motion, constraints)

    # Show post-IK positions
    print(f"\nResults:")
    for c in constraints:
        if c.frame < motion_fixed.shape[0]:
            world_pos, _, _, _ = motion135_to_fk(motion_fixed[c.frame], bone_offsets)
            actual = world_pos[c.joint]
            err = (actual - c.target_xyz).norm().item()
            print(f"  Post-IK: joint {c.joint} at frame {c.frame}: "
                  f"actual={actual.tolist()}, error={err*1000:.2f}mm")

    print(f"\nMax error: {max_error*1000:.2f}mm")

    # Save
    if args.output_npz:
        save_motion_to_npz(motion_fixed, args.output_npz)

    # Also save 135-dim for debugging
    if args.output_npz:
        debug_path = args.output_npz.replace('.npz', '_135.pt')
        torch.save(motion_fixed, debug_path)
        print(f"Saved 135-dim tensor to {debug_path}")


def main():
    parser = argparse.ArgumentParser(description='Position Constraint Demo')
    parser.add_argument('--input_npz', type=str, required=True,
                        help='Input motion NPZ file')
    parser.add_argument('--constraints', type=str, nargs='+', required=True,
                        help='Constraints: "frame=30,joint=20,xyz=1.0,1.5,0.3"')
    parser.add_argument('--output_npz', type=str, default=None,
                        help='Output NPZ path')
    parser.add_argument('--mode', type=str, default='ik',
                        choices=['ik', 'pipeline'],
                        help='Mode: ik (pure IK) or pipeline (ODE + IK)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint (required for pipeline mode)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index')
    args = parser.parse_args()

    if args.mode == 'pipeline' and args.checkpoint is None:
        parser.error("--checkpoint required for pipeline mode")

    if args.mode == 'ik':
        demo_pure_ik(args)
    else:
        print("Pipeline mode requires full model loading. Use --mode ik for standalone demo.")
        print("For pipeline integration, pass position_constraints in the batch dict.")


if __name__ == '__main__':
    main()
