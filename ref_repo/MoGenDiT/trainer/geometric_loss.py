import pdb
import torch
import numpy as np

# from motion_process.motion_representation import *
from Aplus.tools.smpl_light import SMPLight

SKEL_CHAIN = {
    1: [[0, 1], [0, 2], [0, 3]],
    2: [[1, 4], [2, 5], [3, 6]],
    3: [[4, 7], [5, 8], [6, 9]],
    4: [[7, 10], [8, 11], [9, 12], [9, 13], [9, 14]],
    5: [[12, 15], [13, 16], [14, 17]],
    6: [[16, 18], [17, 19]],
    7: [[18, 20], [19, 21]],
    8: [[20, 22], [21, 23]],
}


RIGHT_CHAIN = [2, 5, 8, 11, 14, 17, 19, 21, 23]
LEFT_CHAIN = [1, 4, 7, 10, 13, 16, 18, 20, 22]

SYMMETRIC_JOINT_PAIRS = [
    (1, 2),  # 左髋 ↔ 右髋
    (4, 5),  # 左大腿 ↔ 右大腿
    (7, 8),  # 左小腿 ↔ 右小腿
    (10, 11),  # 左足 ↔ 右足
    (13, 14),  # 左肩 ↔ 右肩
    (16, 17),  # 左大臂 ↔ 右大臂
    (18, 19),  # 左小臂 ↔ 右小臂
    (20, 21),  # 左手 ↔ 右手
    (22, 23),  # 左手指 ↔ 右手指
]

pc_mapping = []
# 分层次的父节点-子节点映射, 用于FK
layered_pc_mapping = {}
#
propagation_matrix = torch.zeros(24, 24)
for k, v in SKEL_CHAIN.items():
    pc_mapping += v
    v = torch.LongTensor(v)
    p_id = np.array(v[:, 0]).tolist()
    c_id = np.array(v[:, 1]).tolist()
    layered_pc_mapping.update({k: [p_id, c_id]})
    propagation_matrix[c_id, p_id] += 1
    propagation_matrix[c_id] += propagation_matrix[p_id]
pc_mapping = torch.LongTensor(pc_mapping)
pc_mapping = [
    np.array(pc_mapping[:, 0]).tolist(),
    np.array(pc_mapping[:, 1]).tolist(),
]


def geometric_loss(R6d, joint, vel, trans, global_pose=False, fps=30):

    R = r6d_to_rotation_matrix(R6d.clone()).reshape(-1, 24, 3, 3)

    # bm = SMPLight()
    # R = forward_kinematics_R(R)
    # R = bm.inverse_kinematics(R)
    #
    # _, joint_from_motion = bm.forward_kinematics(R, calc_joint=True)
    #
    # from sample.gen_test import create_animation
    # create_animation(np.array(joint_from_motion))

    joint = joint.clone().reshape(-1, 24, 3)
    vel = vel.clone().reshape(-1, 24, 3)
    trans = trans.clone().reshape(-1, 3)

    offsets_from_motion = get_skeleton_offsets(
        pose=R, joint=joint, global_pose=global_pose
    )
    init_skeleton_offsets = offsets_from_motion[[0]]
    loss_rigid_body = (
        (offsets_from_motion - init_skeleton_offsets.expand_as(offsets_from_motion))
        ** 2
    ).mean()

    if global_pose:
        joint_from_motion = forward_kinematics_joint_global(
            R=R, skeleton_offsets=init_skeleton_offsets
        )
    else:
        joint_from_motion = forward_kinematics_joint(
            R=R, skeleton_offsets=init_skeleton_offsets
        )

    loss_fk = ((joint - joint_from_motion) ** 2).mean()

    global_joint = joint + trans.unsqueeze(-2)
    global_joint_delta = global_joint[1:] - global_joint[:-1]
    loss_drift = ((global_joint_delta - vel[:-1] / fps) ** 2).mean()

    return loss_rigid_body, loss_fk, loss_drift


def geometric_loss_batch(
    R6d,
    joint,
    vel,
    trans,
    global_pose=False,
    length=None,
    l1_weight=0.0,
    l2_weight=1.0,
):
    b = R6d.shape[0]

    R = r6d_to_rotation_matrix(R6d.clone()).reshape(-1, 24, 3, 3)

    # bm = SMPLight()
    # R = forward_kinematics_R(R)
    # R = bm.inverse_kinematics(R)
    #
    # _, joint_from_motion = bm.forward_kinematics(R, calc_joint=True)
    #
    # from sample.gen_test import create_animation
    # create_animation(np.array(joint_from_motion))

    joint = joint.clone().reshape(-1, 24, 3)
    vel = vel.clone().reshape(b, -1, 24, 3)
    trans = trans.clone().reshape(-1, 3)

    offsets_from_motion = get_skeleton_offsets(
        pose=R, joint=joint, global_pose=global_pose
    ).reshape(b, -1, 24, 3)
    init_skeleton_offsets = offsets_from_motion[:, [0]]
    loss_rigid_body = (
        torch.nn.functional.mse_loss(
            offsets_from_motion,
            init_skeleton_offsets.expand_as(offsets_from_motion),
            reduction="none",
        )
        * l2_weight
    )

    global_joint = (joint + trans.unsqueeze(-2)).reshape(b, -1, 24, 3)
    global_joint_delta = global_joint[:, 1:] - global_joint[:, :-1]
    loss_drift = (
        torch.nn.functional.mse_loss(
            global_joint_delta,
            vel[:, :-1] / fps,
            reduction="none",
        )
        * l2_weight
    )

    # L1 penalty
    if l1_weight > 0.0:
        loss_rigid_body += l1_weight * torch.nn.functional.l1_loss(
            offsets_from_motion,
            init_skeleton_offsets.expand_as(offsets_from_motion),
            reduction="none",
        )
        loss_drift += l1_weight * torch.nn.functional.l1_loss(
            global_joint_delta, vel[:, :-1] / fps, reduction="none"
        )

    if length is not None:
        mask = torch.zeros_like(loss_rigid_body[:, :, :1, :1])
        for i in range(b):
            mask[i, : length[i]] = 1.0
        mask_rigid_body = mask.expand_as(loss_rigid_body)
        loss_rigid_body = (
            loss_rigid_body * mask_rigid_body
        ).sum() / mask_rigid_body.sum()

        mask_drift = mask[:, 1:].expand_as(loss_drift)
        loss_drift = (loss_drift * mask_drift).sum() / mask_drift.sum()
    else:
        loss_rigid_body = loss_rigid_body.mean()
        loss_drift = loss_drift.mean()

    return loss_rigid_body, loss_drift


def r6d_to_rotation_matrix(r6d: torch.Tensor):
    r"""
    Turn 6D vectors into rotation matrices. (torch, batch)

    **Warning:** The two 3D vectors of any 6D vector must be linearly independent.

    :param r6d: 6D vector tensor that can reshape to [batch_size, 6].
    :return: Rotation matrix tensor of shape [batch_size, 3, 3].
    """
    r6d = r6d.reshape(-1, 6)
    column0 = normalize_tensor_eps(r6d[:, 0:3])
    column1 = normalize_tensor_eps(
        r6d[:, 3:6] - (column0 * r6d[:, 3:6]).sum(dim=1, keepdim=True) * column0
    )
    column2 = column0.cross(column1, dim=1)
    r = torch.stack((column0, column1, column2), dim=-1)
    r[torch.isnan(r)] = 0
    return r


def get_skeleton_offsets(pose, joint, global_pose=False):
    pose = pose.clone()
    assert pose.shape[-1] == 3 and pose.shape[-2] == 3
    if not global_pose:
        global_pose = forward_kinematics_R(R=pose).reshape(-1, 24, 3, 3)
    else:
        global_pose = pose.reshape(-1, 24, 3, 3)
    joint = joint.reshape(-1, 24, 3)
    joint_offsets = torch.zeros_like(joint)
    for _, edges in SKEL_CHAIN.items():
        for edge in edges:
            p_idx, c_idx = edge[0], edge[1]
            joint_offsets[:, c_idx] = (
                global_pose[:, p_idx]
                .transpose(-1, -2)
                .matmul((joint[:, c_idx] - joint[:, p_idx]).unsqueeze(-1))
                .squeeze(-1)
            )
    return joint_offsets


def forward_kinematics_R(R):
    for _, mapping in layered_pc_mapping.items():
        p_idx, c_idx = mapping[0], mapping[1]
        R[..., c_idx, :, :] = R[..., p_idx, :, :].matmul(R[..., c_idx, :, :])
    return R


def forward_kinematics_joint(R, skeleton_offsets, trans=None):
    positions = torch.zeros_like(R[..., -1]) + skeleton_offsets.to(R.device)

    if trans is not None:
        positions[..., 0, :] += trans

    # n x 24 x 3 x 4
    Rk = torch.cat([R, positions.unsqueeze(-1)], dim=-1)
    padding = torch.zeros_like(Rk[..., [-1], :])
    padding[..., -1] += 1

    # 构建传递矩阵: [[R, pos],
    #              [0,  1]]
    # n x 24 x 4 x 4
    Rk = torch.cat([Rk, padding], dim=-2)

    # 前向运动学
    for _, mapping in layered_pc_mapping.items():
        p_idx, c_idx = mapping[0], mapping[1]
        Rk[..., c_idx, :, :] = Rk[..., p_idx, :, :].matmul(Rk[..., c_idx, :, :])

    # 获取global的R与pos
    # n x 24 x 3 x 4
    Rk = Rk[..., :-1, :]
    # n x 24 x 3 x 3
    # R = Rk[..., :, :-1]
    # n x 24 x 3
    joint = Rk[..., :, -1]

    return joint


def forward_kinematics_joint_global(R, skeleton_offsets, trans=None):
    positions = torch.zeros_like(R[..., -1]) + skeleton_offsets.to(R.device)
    if trans is not None:
        positions[..., 0, :] += trans
    positions = positions.unsqueeze(-1)  # 转换成列向量

    # 前向运动学
    for _, mapping in layered_pc_mapping.items():
        p_idx, c_idx = mapping[0], mapping[1]
        positions[..., c_idx, :, :] = positions[..., p_idx, :, :] + R[
            ..., p_idx, :, :
        ].matmul(positions[..., c_idx, :, :])

    return positions.squeeze(-1)


def normalize_tensor_eps(x: torch.Tensor, dim=-1, return_norm=False, eps=1e-8):
    r"""
    Normalize a tensor in a specific dimension to unit norm. (torch)

    :param x: Tensor in any shape.
    :param dim: The dimension to be normalized.
    :param return_norm: If True, norm(length) tensor will also be returned.
    :return: Tensor in the same shape. If return_norm is True, norm tensor in shape [*, 1, *] (1 at dim)
             will also be returned (keepdim=True).
    """
    norm = x.norm(dim=dim, keepdim=True) + eps
    normalized_x = x / (norm)
    return normalized_x if not return_norm else (normalized_x, norm)
