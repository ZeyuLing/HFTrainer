import torch

from hftrainer.motion.representation.dart276 import (
    DART276_DIM,
    dart276_to_motion135,
    dart276_to_smpl_params,
    smpl_params_and_joints_to_dart276,
)
from hftrainer.motion.representation.rotation import axis_angle_to_matrix


def test_dart276_smpl_roundtrip_from_consistent_channels():
    torch.manual_seed(7)
    t = 9
    global_orient = torch.randn(t, 3) * 0.2
    body_pose = torch.randn(t, 21, 3) * 0.15
    transl = torch.cumsum(torch.randn(t, 3) * 0.02, dim=0)
    joints = torch.cumsum(torch.randn(t, 22, 3) * 0.01, dim=0)

    m276 = smpl_params_and_joints_to_dart276(
        {
            "global_orient": global_orient,
            "body_pose": body_pose,
            "transl": transl,
        },
        joints,
    )
    assert m276.shape == (t - 1, DART276_DIM)

    smpl, joints_rt = dart276_to_smpl_params(
        m276,
        recover_from_velocity=True,
        equal_length=True,
    )
    m276_rt = smpl_params_and_joints_to_dart276(smpl, joints_rt)
    assert torch.max(torch.abs(m276 - m276_rt)) < 1e-4

    # Axis-angle can wrap, so compare root rotation matrices instead of raw
    # axis-angle vectors.
    assert torch.max(
        torch.abs(axis_angle_to_matrix(global_orient) - axis_angle_to_matrix(smpl["global_orient"]))
    ) < 1e-4
    assert torch.max(torch.abs(transl - smpl["transl"])) < 1e-5
    assert torch.max(torch.abs(joints - joints_rt)) < 1e-5


def test_dart276_to_motion135_row_and_column_shapes():
    torch.manual_seed(11)
    t = 5
    m276 = smpl_params_and_joints_to_dart276(
        {
            "global_orient": torch.randn(t, 3) * 0.1,
            "body_pose": torch.randn(t, 21, 3) * 0.1,
            "transl": torch.randn(t, 3) * 0.01,
        },
        torch.randn(t, 22, 3) * 0.1,
    )
    row = dart276_to_motion135(m276, rotation_convention="row")
    col = dart276_to_motion135(m276, rotation_convention="column")
    assert row.shape == (t, 135)
    assert col.shape == (t, 135)
    assert not torch.allclose(row[:, 3:], col[:, 3:])
