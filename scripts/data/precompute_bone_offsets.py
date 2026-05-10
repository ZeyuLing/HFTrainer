"""Precompute bone offsets from SMPL body model for FK/IK.

Computes the relative bone offset for each joint in the SMPL-22 skeleton:
    offsets[0] = J_template[0]  (root absolute position in T-pose)
    offsets[j] = J_template[j] - J_template[parent[j]]  (relative to parent)

These offsets are used by differentiable FK and IK solvers.

Usage::

    python tools/precompute_bone_offsets.py \\
        --model_path assets/body_models/smplh \\
        --output data/hymotion_m2m_data/bone_offsets_22.pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from hftrainer.datasets.motion.motionhub.transforms.fk_utils import SMPL22_PARENTS


def compute_bone_offsets_from_smpl(
    model_path: str = 'assets/body_models/smplh',
    gender: str = 'neutral',
    num_betas: int = 16,
) -> torch.Tensor:
    """Compute bone offsets from SMPL body model.

    Args:
        model_path: Path to SMPL model directory.
        gender: Gender of the body model.
        num_betas: Number of shape parameters.

    Returns:
        bone_offsets: (22, 3) tensor of bone offsets.
    """
    from hftrainer.models.motion.hymotion_m2m.network.smpl_lite import SmplLite

    model = SmplLite(model_path=model_path, gender=gender, num_betas=num_betas)
    J_template = model.J_template[:22].clone()  # (22, 3)

    offsets = torch.zeros(22, 3)
    offsets[0] = J_template[0]  # root absolute position
    for j in range(1, 22):
        parent = SMPL22_PARENTS[j]
        offsets[j] = J_template[j] - J_template[parent]

    return offsets


def main():
    parser = argparse.ArgumentParser(description='Precompute bone offsets from SMPL body model')
    parser.add_argument(
        '--model_path',
        type=str,
        default='assets/body_models/smplh',
        help='Path to SMPL body model directory',
    )
    parser.add_argument(
        '--gender',
        type=str,
        default='neutral',
        help='Gender of the body model',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/hymotion_m2m_data/bone_offsets_22.pt',
        help='Output path for bone offsets',
    )
    args = parser.parse_args()

    offsets = compute_bone_offsets_from_smpl(args.model_path, args.gender)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(offsets, args.output)
    print(f'Saved bone offsets to {args.output}')
    print(f'Shape: {offsets.shape}')
    print(f'Root offset (T-pose pelvis): {offsets[0].tolist()}')

    # Print bone lengths
    for j in range(1, 22):
        length = offsets[j].norm().item()
        from hftrainer.models.motion.hymotion_m2m.network.smpl_lite import SMPLX_NUM2JOINT
        name = SMPLX_NUM2JOINT.get(j, f'Joint{j}')
        parent = SMPL22_PARENTS[j]
        parent_name = SMPLX_NUM2JOINT.get(parent, f'Joint{parent}')
        print(f'  {name}({j}) <- {parent_name}({parent}): length={length:.4f}m')


if __name__ == '__main__':
    main()
