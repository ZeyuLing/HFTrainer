import torch
import numpy as np
from articulate.math.angular import (
    r6d_norm,
    r6d_to_rotation_matrix,
    rotation_matrix_to_r6d,
)
from .rotation_conversions import euler_angles_to_matrix
from animo.skeleton.smpl_body import AnimoSMPLBody
from .utils import get_ego_gv
from .motion_degradation import GlobalMotionDegradation

# from .kinematics import *
import random
import time

# body_model = AnimoSMPLBody()


def random_index(data_len: int, sampling_rate=1.0, seed: int = None) -> list:
    """
    随机采样索引的函数

    参数:
    sample_size (int): 样本数量
    sampling_rate (float): 采样率，范围在(0, 1]
    seed (int or None): 随机数生成的种子，如果为None则不设置种子

    返回:
    list: 随机采样的索引列表
    """
    if not (0 < sampling_rate <= 1):
        raise ValueError("采样率必须在(0, 1]范围内")

    # 设置随机数种子
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(int(time.time()))

    # 计算需要采样的样本数量
    num_samples_to_select = int(data_len * sampling_rate)

    # 生成样本索引的随机排列
    all_indices = np.arange(data_len)
    np.random.shuffle(all_indices)

    # 从随机排列的索引中选择需要的数量
    selected_indices = all_indices[:num_samples_to_select]

    return selected_indices


class HM263XRep:
    def __init__(self, keep_hand=False):
        self.n_joint = 22
        if keep_hand:
            self.n_joint = self.n_joint
        self.data_dim = (
            self.n_joint * 6 + self.n_joint * 3 + self.n_joint * 1
        )  # pose(self.n_joint*6) + joint(n_ric*3) + stationary(n_ric*1)
        self.padding_dim = 263 - self.data_dim
        self.SCALE_JOINTS = 2  # 用于归一化关节位置的缩放因子
        self.SCALE_VEL = 3  # 用于归一化关节速度的缩放因子
        self.SCALE_TRANS = 1

    def encode(self, pose: torch.Tensor, joint: torch.Tensor, stationary: torch.Tensor):
        """
        pose: (T, self.n_joint, 6) 6d representation
        joint: (T, self.n_joint/22, 3) 3d joint positions
        vel: (T, self.n_joint/22, 3) 3d joint velocities
        """
        assert (
            pose.ndim == 3 and pose.shape[1] == self.n_joint and pose.shape[2] == 6
        ), f"pose shape {pose.shape} is not correct"
        assert (
            joint.ndim == 3 and joint.shape[1] == self.n_joint and joint.shape[2] == 3
        ), f"joint shape {joint.shape} is not correct"
        assert (
            stationary.ndim == 2 and stationary.shape[1] == self.n_joint
        ), f"stationary shape {stationary.shape} is not correct"
        pose = pose.flatten(1)
        joint = joint.flatten(1) / self.SCALE_JOINTS
        stationary = stationary.flatten(1)

        hm263x_motion = torch.cat([pose, joint, stationary], dim=-1)
        padding = torch.zeros(
            (hm263x_motion.shape[0], self.padding_dim),
            dtype=pose.dtype,
            device=hm263x_motion.device,
        )
        hm263x_motion = torch.cat([hm263x_motion, padding], dim=-1)

        return hm263x_motion

    def decode(self, hm263x_motion: torch.Tensor):
        """
        返回:
          pose: (T, self.n_joint, 6) 6d 表示
          joint: (T, self.n_joint/22, 3) 3d joint positions
          vel: (T, self.n_joint/22, 3) 3d joint velocities
        """
        if hm263x_motion.dim() != 2:
            raise ValueError(
                f"hm263x_motion must be 2D batch, got {hm263x_motion.shape}"
            )

        T = hm263x_motion.shape[0]
        # 直接分割出非 padding 部分
        x = hm263x_motion[:, : self.data_dim]  # (T, data_dim)

        # 还原顺序：pose(self.n_joint*6) -> joint(n_ric*3) -> vel(n_ric*3)
        pose_flat = x[:, : self.n_joint * 6]  # (T, 144)
        joint_flat = x[:, self.n_joint * 6 : self.n_joint * 6 + self.n_joint * 3]
        stationary = x[
            :, self.n_joint * 6 + self.n_joint * 3 : self.data_dim
        ]  # (T, n_ric*3)

        # 重构形状
        pose = pose_flat.view(T, self.n_joint, 6)
        pose = r6d_norm(pose)
        joint = joint_flat.view(T, self.n_joint, 3) * self.SCALE_JOINTS

        return pose, joint, stationary

    def normalization(self, hm263x_motion: torch.Tensor, ref_idx=0):
        """
        对数据进行标准化, 保障ref_idx的motion数据帧面朝z轴, 且位于x-z平面原点
        ref_idx默认为0 (起始帧)
        """
        if hm263x_motion.dim() != 2:
            raise ValueError(
                f"hm263x_motion must be 2D batch, got {hm263x_motion.shape}"
            )

        if ref_idx < 0:
            ref_idx = len(hm263x_motion) + ref_idx

        T = hm263x_motion.shape[0]

        root_ori_6d = hm263x_motion[:, : 1 * 6]  # (T, 144)
        root_ori = r6d_to_rotation_matrix(root_ori_6d)
        R_ego_gv_inv = get_ego_gv(root_ori[ref_idx]).transpose(-2, -1)
        root_ori = R_ego_gv_inv.matmul(root_ori)
        root_ori_6d = rotation_matrix_to_r6d(root_ori)

        joint = hm263x_motion[
            :, self.n_joint * 6 : self.n_joint * 6 + self.n_joint * 3
        ].reshape(-1, self.n_joint, 3)
        joint[:, :, [0, 2]] -= joint[ref_idx : ref_idx + 1, :1, [0, 2]]
        joint = R_ego_gv_inv.matmul(joint.unsqueeze(-1)).view_as(joint)
        # vel_flat = x[:, self.n_joint * 6 + self.n_joint * 3: ]  # (T, n_ric*3)

        # 覆写
        hm263x_motion[:, : 1 * 6] = root_ori_6d.flatten(1)
        hm263x_motion[:, self.n_joint * 6 : self.n_joint * 6 + self.n_joint * 3] = (
            joint.flatten(1)
        )

        return hm263x_motion


# class Motion291Rep:
#     def __init__(self, keep_hand=True, global_pose=False, fps=30, use_vel=True):
#         self.n_joint = 22
#         if keep_hand:
#             self.n_joint = self.n_joint
#         self.global_pose = global_pose

#         d_pose = self.n_joint * 6
#         d_joint = self.n_joint * 3
#         d_vel = self.n_joint * 3
#         d_trans = 3

#         self.fps = fps
#         self.SCALE_VEL = 3  # 用于归一化关节速度的缩放因子
#         self.use_vel = use_vel

#         self.data_dim = (
#             d_pose + d_joint + d_vel + d_trans
#         )  # pose(self.n_joint*6) + local_joint(n_ric*3) + vel(n_ric*3) + trans(3)
#         # self.padding_dim = 300 - self.data_dim
#         self.pose_mask, self.joint_mask, self.vel_mask, self.trans_mask = (
#             torch.zeros(self.data_dim, dtype=torch.bool) for _ in range(4)
#         )
#         self.pose_mask[:d_pose] = True
#         self.joint_mask[d_pose : d_pose + d_joint] = True
#         self.vel_mask[d_pose + d_joint : d_pose + d_joint + d_vel] = True
#         self.trans_mask[
#             d_pose + d_joint + d_vel : d_pose + d_joint + d_vel + d_trans
#         ] = True

#         self.mask_dict = {
#             "pose": self.pose_mask,
#             "joint": self.joint_mask,
#             "vel": self.vel_mask,
#             "trans": self.trans_mask,
#         }
#         self.body_model = SMPLight()

#     def encode(self, pose: torch.Tensor, joint: torch.Tensor, vel: torch.Tensor):
#         """
#         pose: (T, self.n_joint, 3, 3) rotation matrix representation
#         joint: (T, self.n_joint/22, 3) 3d global joint positions
#         vel: (T, self.n_joint/22, 3) 3d joint velocities
#         """
#         assert (
#             pose.ndim == 4
#             and pose.shape[1] == self.n_joint
#             and pose.shape[2] == 3
#             and pose.shape[3] == 3
#         ), f"pose shape {pose.shape} is not correct"
#         assert (
#             joint.ndim == 3 and joint.shape[1] == self.n_joint and joint.shape[2] == 3
#         ), f"joint shape {joint.shape} is not correct"
#         assert (
#             vel.ndim == 3 and vel.shape[1] == self.n_joint
#         ), f"velocity shape {vel.shape} is not correct"

#         if self.global_pose:
#             pose = self.body_model.forward_kinematics(pose)
#         pose = rotation_matrix_to_r6d(pose).reshape(-1, self.n_joint, 6)

#         if not self.use_vel:
#             vel = torch.zeros_like(vel)

#         trans = joint[:, [0]]
#         joint -= trans
#         pose = pose.flatten(1)
#         joint = joint.flatten(1)
#         vel = vel.flatten(1) * self.fps / self.SCALE_VEL
#         trans = trans.flatten(1)

#         motion = torch.cat([pose, joint, vel, trans], dim=-1)

#         return motion

#     def encode_batch(self, pose: torch.Tensor, joint: torch.Tensor, vel: torch.Tensor):
#         """
#         处理 batch 数据的编码函数，输入增加 batch 维度
#         pose: (B, T, self.n_joint, 3, 3) 旋转矩阵表示，B 为 batch size，T 为时间步
#         joint: (B, T, n_ric, 3) 3D 全局关节位置（n_ric 为 self.n_joint 或 22）
#         vel: (B, T, n_ric, 3) 3D 关节速度
#         """
#         # 维度和形状检查（增加对 batch 维度的支持）
#         assert (
#             pose.ndim == 5
#             and pose.shape[2] == self.n_joint
#             and pose.shape[3] == 3
#             and pose.shape[4] == 3
#         ), f"pose 形状 {pose.shape} 不正确，应为 (B, T, self.n_joint, 3, 3)"
#         assert (
#             joint.ndim == 4 and joint.shape[2] == self.n_joint and joint.shape[3] == 3
#         ), f"joint 形状 {joint.shape} 不正确，应为 (B, T, {self.n_joint}, 3)"
#         assert (
#             vel.ndim == 4 and vel.shape[2] == self.n_joint and vel.shape[3] == 3
#         ), f"vel 形状 {vel.shape} 不正确，应为 (B, T, {self.n_joint}, 3)"

#         # 如果需要全局姿态，对每个 batch 中的每个时间步计算前向运动学
#         if self.global_pose:
#             # 注意：需确保 body_model.forward_kinematics 支持 batch 输入
#             # 若原函数不支持，可能需要遍历 batch 或调整实现
#             pose = self.body_model.forward_kinematics(pose)

#         # 将旋转矩阵转换为 6D 表示（保持 batch 和时间维度）
#         # rotation_matrix_to_r6d 需支持 (B, T, self.n_joint, 3, 3) 输入，输出 (B, T, self.n_joint, 6)
#         pose = rotation_matrix_to_r6d(pose).reshape(
#             pose.shape[0], pose.shape[1], self.n_joint, 6
#         )

#         if not self.use_vel:
#             vel = torch.zeros_like(vel)

#         # 提取根关节平移（batch 维度下取每个样本的根关节）
#         trans = joint[:, :, [0]]  # 形状: (B, T, 1, 3)

#         # 关节位置减去根关节平移（消除全局位置偏移）
#         joint = joint - trans  # 形状保持 (B, T, n_ric, 3)

#         # 展平特征维度（保留 batch 和时间维度）
#         pose = pose.flatten(2)  # (B, T, self.n_joint*6) = (B, T, 144)
#         joint = joint.flatten(2)  # (B, T, n_ric*3)
#         vel = vel.flatten(2) * self.fps / self.SCALE_VEL  # (B, T, n_ric*3)
#         trans = trans.flatten(2)  # (B, T, 1*3) = (B, T, 3)

#         # 在特征维度拼接所有信息
#         motion = torch.cat([pose, joint, vel, trans], dim=-1)  # 形状: (B, T, 总特征数)

#         return motion

#     def decode(self, motion: torch.Tensor):
#         """
#         返回:
#           pose: (T, self.n_joint, 6) 6d 表示
#           joint: (T, self.n_joint/22, 3) 3d joint positions
#           vel: (T, self.n_joint/22, 3) 3d joint velocities
#         """
#         if motion.dim() != 2:
#             raise ValueError(f"hm263x_motion must be 2D batch, got {motion.shape}")

#         T = motion.shape[0]
#         # 直接分割出非 padding 部分
#         x = motion[:, : self.data_dim]  # (T, data_dim)

#         # 还原顺序：pose(self.n_joint*6) -> joint(n_ric*3) -> vel(n_ric*3)
#         pose_flat = x[:, self.pose_mask.to(x.device)]
#         joint_flat = x[:, self.joint_mask.to(x.device)]
#         vel_flat = x[:, self.vel_mask.to(x.device)] * self.SCALE_VEL / self.fps
#         trans_flat = x[:, self.trans_mask.to(x.device)]

#         # 重构形状
#         pose = pose_flat.view(T, self.n_joint, 6)
#         pose = r6d_to_rotation_matrix(pose).reshape(T, self.n_joint, 3, 3)
#         if self.global_pose:
#             pose = self.body_model.inverse_kinematics(pose)
#         joint = joint_flat.view(T, self.n_joint, 3)
#         vel = vel_flat.view(T, self.n_joint, 3)
#         trans = trans_flat.view(T, 3)
#         joint += trans.unsqueeze(1)

#         return pose, joint, vel

#     def normalization(self, motion: torch.Tensor, ref_idx=0):
#         motion = motion.clone()
#         """
#         对数据进行标准化, 保障ref_idx的motion数据帧面朝z轴, 且位于x-z平面原点
#         ref_idx默认为0 (起始帧)
#         """
#         if motion.dim() != 2:
#             raise ValueError(f"hm263x_motion must be 2D batch, got {motion.shape}")

#         if ref_idx < 0:
#             ref_idx = len(motion) + ref_idx

#         T = motion.shape[0]

#         # 提提取并处理pose
#         pose = r6d_to_rotation_matrix(
#             motion[:, self.pose_mask.to(motion.device)]
#         ).reshape(T, self.n_joint, 3, 3)
#         R_ego_gv_inv = get_ego_gv(pose[ref_idx, 0]).transpose(-2, -1)
#         if not self.global_pose:
#             pose[:, 0] = R_ego_gv_inv.matmul(pose[:, 0])
#         else:
#             pose = R_ego_gv_inv.matmul(pose)

#         pose = rotation_matrix_to_r6d(pose).reshape(-1, self.n_joint * 6)

#         # 提提取并处理joint vel trans
#         joint = motion[:, self.joint_mask.to(motion.device)].reshape(
#             -1, self.n_joint, 3
#         )
#         vel = motion[:, self.vel_mask.to(motion.device)].reshape(-1, self.n_joint, 3)
#         trans = motion[:, self.trans_mask.to(motion.device)].reshape(-1, 1, 3)

#         global_joint = joint + trans
#         global_joint[:, :, [0, 2]] -= global_joint[ref_idx : ref_idx + 1, :1, [0, 2]]
#         global_joint = R_ego_gv_inv.matmul(global_joint.unsqueeze(-1)).view_as(
#             global_joint
#         )
#         init_h = global_joint[:60, :, 1].min()
#         global_joint[:, :, 1] -= init_h

#         vel = R_ego_gv_inv.matmul(vel.unsqueeze(-1))
#         trans = global_joint[:, 0]
#         joint = global_joint - global_joint[:, [0]]

#         # 覆写
#         motion[:, self.pose_mask] = pose.flatten(1)
#         motion[:, self.joint_mask] = joint.flatten(1)
#         motion[:, self.vel_mask] = vel.flatten(1)
#         motion[:, self.trans_mask] = trans.flatten(1)

#         return motion

#     def pre_stitch(
#         self,
#         motion: torch.Tensor,
#         ref_motion: torch.Tensor,
#         ref_idx=0,
#         reset_height=False,
#         sync_skeleton=False,
#     ):
#         motion = motion.clone()
#         """
#         对数据进行标准化, 保障ref_idx的motion数据帧面朝z轴, 且位于x-z平面原点
#         ref_idx默认为0 (起始帧)
#         """
#         if motion.dim() != 2:
#             raise ValueError(f"hm263x_motion must be 2D batch, got {motion.shape}")

#         if ref_idx < 0:
#             ref_idx = len(motion) + ref_idx

#         if reset_height:
#             trans_reset_axis = [0, 2]
#         else:
#             trans_reset_axis = [0, 1, 2]

#         T = motion.shape[0]

#         # 提提取并处理pose
#         pose = r6d_to_rotation_matrix(
#             motion[:, self.pose_mask.to(motion.device)]
#         ).reshape(T, self.n_joint, 3, 3)

#         ref_motion = ref_motion.to(motion.device)
#         # decode出来的pose固定是local pose
#         ref_pose, ref_joint, _ = self.decode(ref_motion)

#         # 把motion的骨架与ref的骨架进行同步 并重新调整高度
#         if sync_skeleton:
#             origin_joint = motion[:, self.joint_mask.to(motion.device)].reshape(
#                 -1, self.n_joint, 3
#             )
#             root_h_origin = (
#                 origin_joint[0, 0, 1] - origin_joint[0, :, 1].min()
#             )  # 根节点原始高度
#             ref_skel_offsets = get_skeleton_offsets(
#                 pose=ref_pose, joint=ref_joint, global_pose=False
#             )
#             # mask取出的pose 有可能是global pose 也有可能是local pose
#             if self.global_pose:
#                 fk = forward_kinematics_joint_global
#             else:
#                 fk = forward_kinematics_joint
#             joint_sync = fk(R=pose.clone(), skeleton_offsets=ref_skel_offsets)
#             motion[:, self.joint_mask.to(motion.device)] = joint_sync.reshape(
#                 -1, self.n_joint * 3
#             )
#             root_h_sync = (
#                 joint_sync[0, 0, 1] - joint_sync[0, :, 1].min()
#             )  # 根节点原始高度
#             motion[:, self.trans_mask.to(motion.device)][1] += (
#                 root_h_sync - root_h_origin
#             )

#         # 朝向对齐
#         R_ego_gv_inv = get_ego_gv(pose[ref_idx, 0]).transpose(-2, -1)
#         R_ego_gv_ref = get_ego_gv(ref_pose[0, 0])
#         ori_transform = R_ego_gv_ref.matmul(R_ego_gv_inv)

#         if not self.global_pose:
#             pose[:, 0] = ori_transform.matmul(pose[:, 0])
#         else:
#             pose = ori_transform.matmul(pose)

#         pose = rotation_matrix_to_r6d(pose).reshape(-1, self.n_joint * 6)

#         # 提提取并处理joint vel trans
#         joint = motion[:, self.joint_mask.to(motion.device)].reshape(
#             -1, self.n_joint, 3
#         )
#         vel = motion[:, self.vel_mask.to(motion.device)].reshape(-1, self.n_joint, 3)
#         trans = motion[:, self.trans_mask.to(motion.device)].reshape(-1, 1, 3)

#         global_joint = joint + trans
#         global_joint[:, :, trans_reset_axis] -= global_joint[
#             ref_idx : ref_idx + 1, :1, trans_reset_axis
#         ].clone()
#         global_joint = ori_transform.matmul(global_joint.unsqueeze(-1)).squeeze(-1)

#         # 修改trans对齐逻辑 此时global_joint的trans_reset_axis已经被重置
#         # 原版是与ref trans对齐 是ref_joint_idx = 0的特例
#         # trans_ref = ref_motion[:, self.trans_mask.to(motion.device)].reshape(1, 1, 3)
#         # global_joint[:, :, trans_reset_axis] += trans_ref[:, :, trans_reset_axis]

#         # 新版加入ref_joint_idx设置，让衔接后ref_joint_idx关节位置对齐
#         trans_ref = ref_motion[:, self.trans_mask.to(motion.device)].reshape(1, 1, 3)
#         lowest_ref_joint_idx = torch.argmin(
#             ref_joint[0, :, 1]
#         )  # 取ref的最低点作为对齐关节
#         ref_root_2_ref_joint = ref_motion[:, self.joint_mask.to(motion.device)].reshape(
#             -1, self.n_joint, 3
#         )[:1, [lowest_ref_joint_idx]]
#         ref_joint_2_motion_root = (
#             global_joint[:1, [0]] - global_joint[:1, [lowest_ref_joint_idx]]
#         )
#         trans_ref += ref_root_2_ref_joint + ref_joint_2_motion_root

#         global_joint[:, :, trans_reset_axis] += trans_ref[:, :, trans_reset_axis]

#         if reset_height:
#             init_height = global_joint[0, 0, 1] - global_joint[0, :, 1].min()
#             ground_ref = ref_joint[
#                 0, :, 1
#             ].min()  # 假设ref的最低点触地 其值视为地面高度
#             global_joint[:, :, 1] -= global_joint[:1, [0], 1]  # motion初始高度置0
#             global_joint[:, :, 1] += init_height + ground_ref  # 地面为ref时的高度

#         vel = ori_transform.matmul(vel.unsqueeze(-1))
#         trans = global_joint[:, 0]
#         joint = global_joint - global_joint[:, [0]]

#         # 覆写
#         motion[:, self.pose_mask] = pose.flatten(1)
#         motion[:, self.joint_mask] = joint.flatten(1)
#         motion[:, self.vel_mask] = vel.flatten(1)
#         motion[:, self.trans_mask] = trans.flatten(1)

#         return motion

#     @torch.no_grad()
#     def geometric_degradation_batch(
#         self,
#         motion: torch.Tensor,
#         keyframe_mask=None,
#         length=None,
#         bool_length_mask=None,
#     ):
#         b = motion.shape[0]
#         seq_len = motion.shape[1]
#         motion = motion.clone()
#         motion_origin = motion.clone()

#         # 提提取并处理pose
#         pose = r6d_to_rotation_matrix(
#             motion[..., self.pose_mask.to(motion.device)]
#         ).reshape(-1, self.n_joint, 3, 3)

#         pose = pose.reshape(b, -1, self.n_joint, 3, 3)

#         # 提提取并处理joint vel trans
#         joint = motion[..., self.joint_mask.to(motion.device)]

#         trans = motion[..., self.trans_mask.to(motion.device)].reshape(b, -1, 3)

#         degradation_prob = torch.bernoulli(
#             torch.full((b, 1, 1, 1), 0.5)
#         )  # 50%的数据进行degradation

#         degradation_scales = torch.rand((b, 2, 1, 1)) * degradation_prob
#         # degradation_scales = torch.ones((b, 2, 1, 1)) * degradation_prob
#         degradation_scales = degradation_scales.reshape(b * 2, 1, 1)

#         (
#             degradation_scale_trans,
#             degradation_scale_pose,
#         ) = degradation_scales.chunk(2, dim=0)

#         rotation_degration = (
#             ((torch.rand(b, seq_len, self.n_joint, 3)) - 0.5)
#             * 2
#             * (np.pi / 180)
#             * 120.0
#             * degradation_scale_pose.unsqueeze(-1)
#         )
#         # 0-50% 的概率出现关节跳变
#         rotation_degration_mask = torch.bernoulli(
#             torch.ones(b, 1, 1, 1).repeat(1, seq_len, self.n_joint, 1)
#             * torch.rand(b, 1, 1, 1)
#             * 0.5
#         )
#         rotation_degration *= rotation_degration_mask

#         rotation_degration_matrix = euler_angles_to_matrix(
#             rotation_degration.reshape(-1, 3).to(motion.device), convention="XYZ"
#         ).reshape(
#             b, -1, self.n_joint, 3, 3
#         )  # b x self.n_joint x 3 x 3

#         # 高斯过程漂移
#         gaussian_delta_xz_drift = (
#             (torch.randn(b, 1, 2) + torch.randn(b, seq_len, 2))
#             * 0.05
#             * degradation_scale_trans
#         )
#         gaussian_delta_y_drift = (
#             torch.randn(b, seq_len, 1) * 0.01 * degradation_scale_trans
#         )
#         if length is not None:

#             zero_mask = 1 - get_temporal_mask(
#                 motion=gaussian_delta_xz_drift, length=length, mode="random_phrase"
#             )
#             gaussian_delta_xz_drift *= zero_mask

#             zero_mask_y = 1 - get_temporal_mask(
#                 motion=gaussian_delta_y_drift, length=length, mode="random_phrase"
#             )
#             gaussian_delta_y_drift *= zero_mask_y.to(gaussian_delta_y_drift.device)

#         gaussian_delta_xz_drift = gaussian_delta_xz_drift[:, 1:]
#         gaussian_delta_y_drift = gaussian_delta_y_drift[:, 1:]

#         # joint受pose影响
#         offsets_from_motion = get_skeleton_offsets(
#             pose=pose, joint=joint, global_pose=self.global_pose
#         ).reshape(b, -1, self.n_joint, 3)
#         init_skeleton_offsets = offsets_from_motion[:, [0]]
#         pose = pose.matmul(rotation_degration_matrix)
#         if self.global_pose:
#             joint = forward_kinematics_joint_global(
#                 R=pose.reshape(b, -1, self.n_joint, 3, 3),
#                 skeleton_offsets=init_skeleton_offsets.reshape(b, -1, self.n_joint, 3),
#             ).reshape(b, -1, self.n_joint * 3)
#         else:
#             joint = forward_kinematics_joint(
#                 R=pose.reshape(b, -1, self.n_joint, 3, 3),
#                 skeleton_offsets=init_skeleton_offsets.reshape(b, -1, self.n_joint, 3),
#             ).reshape(b, -1, self.n_joint * 3)

#         pose = rotation_matrix_to_r6d(pose).reshape(b, -1, self.n_joint * 6)
#         # trans倍率扰动
#         delta_trans = trans[:, 1:] - trans[:, :-1]
#         delta_trans *= 1 + (
#             torch.rand(b, 1, 3).to(motion.device) * 0.6 - 0.3
#         ) * degradation_scale_trans.to(motion.device)
#         delta_trans[:, :, [0, 2]] += gaussian_delta_xz_drift.to(motion.device)
#         delta_trans[:, :, [1]] += gaussian_delta_y_drift.to(motion.device)
#         trans[:, 1:] = trans[:, :1] + torch.cumsum(delta_trans, dim=1)

#         # trans[:, 1:, [0, 2]] += gaussian_xz_drift.to(motion.device)
#         # trans[:, 1:, [1]] += gaussian_y_drift.to(motion.device)
#         trans = trans.reshape(b, -1, 3)

#         motion[..., self.pose_mask] = pose
#         motion[..., self.joint_mask] = joint
#         motion[..., self.trans_mask] = trans

#         # vel受joint与trans影响
#         if self.use_vel:
#             vel = (joint[:, 1:] - joint[:, :-1]).reshape(b, -1, self.n_joint, 3)
#             vel += (trans[:, 1:] - trans[:, :-1]).unsqueeze(2)
#             vel = torch.cat([vel, vel[:, -1:]], dim=1)
#             vel = vel * self.fps / self.SCALE_VEL

#             motion[..., self.vel_mask] = vel.reshape(b, -1, self.n_joint * 3)

#         if keyframe_mask is not None:
#             keyframe_mask_bool = keyframe_mask == 1
#             motion[keyframe_mask_bool] = motion_origin[keyframe_mask_bool]

#         if bool_length_mask is not None:
#             motion[~bool_length_mask] *= 0.0

#         return motion

#     def get_component(self, motion, component_name: str):
#         assert component_name in ["pose", "joint", "vel", "trans"]
#         if component_name == "vel":
#             scale = self.SCALE_VEL
#         else:
#             scale = 1

#         return motion[..., self.mask_dict[component_name].to(motion.device)] * scale


class OccamMotionRep:
    def __init__(self, keep_hand=False, global_pose=True, fps=30):
        self.n_joint = 22
        assert keep_hand == False, "OccamMotionRep 手部关节支持还在开发中"
        self.global_pose = global_pose

        d_pose = self.n_joint * 6
        d_joint = self.n_joint * 3
        d_trans = 3

        self.fps = fps

        self.data_dim = d_pose + d_joint + d_trans
        self.pose_mask, self.joint_mask, self.trans_mask = (
            torch.zeros(self.data_dim, dtype=torch.bool) for _ in range(3)
        )
        self.pose_mask[:d_pose] = True
        self.pose_wo_hand_mask = self.pose_mask.clone()
        self.pose_wo_hand_mask[20*6:] = False
        self.joint_mask[d_pose : d_pose + d_joint] = True
        self.trans_mask[d_pose + d_joint : d_pose + d_joint + d_trans] = True

        self.mask_dict = {
            "pose": self.pose_mask,
            "joint": self.joint_mask,
            "trans": self.trans_mask,
        }
        self.body_model = AnimoSMPLBody()
        self.degradation = GlobalMotionDegradation()

    def encode(self, pose: torch.Tensor, joint: torch.Tensor, trans: torch.Tensor):
        """
        pose: (T, self.n_joint, 3, 3) rotation matrix representation
        joint: (T, self.n_joint/22, 3) 3d global joint positions
        trans: (T, 3) 3d root joint translations
        """
        assert (
            pose.ndim == 4
            and pose.shape[1] == self.n_joint
            and pose.shape[2] == 3
            and pose.shape[3] == 3
        ), f"pose shape {pose.shape} is not correct"
        assert (
            joint.ndim == 3 and joint.shape[1] == self.n_joint and joint.shape[2] == 3
        ), f"joint shape {joint.shape} is not correct"
        assert (
            trans.ndim == 2 and trans.shape[1] == 3
        ), f"trans shape {trans.shape} is not correct"

        if self.global_pose:
            pose = self.body_model.forward_kinematics(pose)
        pose = rotation_matrix_to_r6d(pose).reshape(-1, self.n_joint, 6)

        pose = pose.flatten(1)
        joint = joint.flatten(1)
        trans = trans.flatten(1)

        motion = torch.cat([pose, joint, trans], dim=-1)

        return motion

    def encode_batch(
        self, pose: torch.Tensor, joint: torch.Tensor, trans: torch.Tensor
    ):
        """
        处理 batch 数据的编码函数，输入增加 batch 维度
        pose: (B, T, self.n_joint, 3, 3) 旋转矩阵表示，B 为 batch size，T 为时间步
        joint: (B, T, n_ric, 3) 3D 全局关节位置（n_ric 为 self.n_joint 或 22）
        """
        # 维度和形状检查（增加对 batch 维度的支持）
        assert (
            pose.ndim == 5
            and pose.shape[2] == self.n_joint
            and pose.shape[3] == 3
            and pose.shape[4] == 3
        ), f"pose 形状 {pose.shape} 不正确，应为 (B, T, {self.n_joint}, 3, 3)"
        assert (
            joint.ndim == 4 and joint.shape[2] == self.n_joint and joint.shape[3] == 3
        ), f"joint 形状 {joint.shape} 不正确，应为 (B, T, {self.n_joint}, 3)"
        assert (
            trans.ndim == 3 and trans.shape[2] == 3
        ), f"trans 形状 {trans.shape} 不正确，应为 (B, T, 1, 3)"

        # 如果需要全局姿态，对每个 batch 中的每个时间步计算前向运动学
        if self.global_pose:
            # 注意：需确保 body_model.forward_kinematics 支持 batch 输入
            # 若原函数不支持，可能需要遍历 batch 或调整实现
            pose = self.body_model.forward_kinematics(pose)

        # 将旋转矩阵转换为 6D 表示（保持 batch 和时间维度）
        # rotation_matrix_to_r6d 需支持 (B, T, self.n_joint, 3, 3) 输入，输出 (B, T, self.n_joint, 6)
        pose = rotation_matrix_to_r6d(pose).reshape(
            pose.shape[0], pose.shape[1], self.n_joint, 6
        )

        # 展平特征维度（保留 batch 和时间维度）
        pose = pose.flatten(2)  # (B, T, self.n_joint*6) = (B, T, 144)
        joint = joint.flatten(2)  # (B, T, n_ric*3)
        trans = trans.flatten(2)  # (B, T, 1*3) = (B, T, 3)

        # 在特征维度拼接所有信息
        motion = torch.cat([pose, joint, trans], dim=-1)  # 形状: (B, T, 总特征数)

        return motion

    def decode(self, motion: torch.Tensor):
        """
        返回:
          pose: (T, self.n_joint, 6) 6d 表示
          joint: (T, self.n_joint/22, 3) 3d joint positions
          vel: (T, self.n_joint/22, 3) 3d joint velocities
        """
        if motion.dim() != 2:
            raise ValueError(f"hm263x_motion must be 2D batch, got {motion.shape}")

        T = motion.shape[0]
        # 直接分割出非 padding 部分
        x = motion[:, : self.data_dim]  # (T, data_dim)

        # 还原顺序：pose(self.n_joint*6) -> joint(n_ric*3) -> vel(n_ric*3)
        pose_flat = x[:, self.pose_mask.to(x.device)]
        joint_flat = x[:, self.joint_mask.to(x.device)]
        trans_flat = x[:, self.trans_mask.to(x.device)]

        # 重构形状
        pose = pose_flat.view(T, self.n_joint, 6)
        pose = r6d_to_rotation_matrix(pose).reshape(T, self.n_joint, 3, 3)
        if self.global_pose:
            pose = self.body_model.inverse_kinematics(pose)
        joint = joint_flat.view(T, self.n_joint, 3)
        trans = trans_flat.view(T, 3)

        return pose, joint, trans

    def normalization(self, motion: torch.Tensor, ref_idx=0, height_reset=False):
        motion = motion.clone()
        """
        对数据进行标准化, 保障ref_idx的motion数据帧面朝z轴, 且位于x-z平面原点
        ref_idx默认为0 (起始帧)
        """
        if motion.dim() != 2:
            raise ValueError(f"hm263x_motion must be 2D batch, got {motion.shape}")

        if ref_idx < 0:
            ref_idx = len(motion) + ref_idx

        T = motion.shape[0]

        # 提提取并处理pose
        pose = r6d_to_rotation_matrix(
            motion[:, self.pose_mask.to(motion.device)]
        ).reshape(T, self.n_joint, 3, 3)
        R_ego_gv_inv = get_ego_gv(pose[ref_idx, 0]).transpose(-2, -1)
        if not self.global_pose:
            pose[:, 0] = R_ego_gv_inv.matmul(pose[:, 0])
        else:
            pose = R_ego_gv_inv.matmul(pose)

        pose = rotation_matrix_to_r6d(pose).reshape(-1, self.n_joint * 6)

        # 提提取并处理joint vel trans
        joint = motion[:, self.joint_mask.to(motion.device)].reshape(
            -1, self.n_joint, 3
        )
        trans = motion[:, self.trans_mask.to(motion.device)].reshape(-1, 1, 3)

        global_joint = joint + trans
        global_joint[:, :, [0, 2]] -= global_joint[ref_idx : ref_idx + 1, :1, [0, 2]]
        global_joint = R_ego_gv_inv.matmul(global_joint.unsqueeze(-1)).view_as(
            global_joint
        )
        if height_reset:
            # 重置高度 让最低点为0
            init_h = global_joint[:60, :, 1].min()
            global_joint[:, :, 1] -= init_h

        trans = global_joint[:, 0]
        joint = global_joint - global_joint[:, [0]]

        # 覆写
        motion[:, self.pose_mask] = pose.flatten(1)
        motion[:, self.joint_mask] = joint.flatten(1)
        motion[:, self.trans_mask] = trans.flatten(1)

        return motion

    def pre_stitch(
        self,
        motion: torch.Tensor,
        ref_motion: torch.Tensor,
        ref_idx=0,
        stitch_joint_idx=None,
        reset_height=False,
        sync_skeleton=False,
    ):
        motion = motion.clone()
        """
        将数据与参考帧进行预缝合
        """
        if motion.dim() != 2:
            raise ValueError(f"hm263x_motion must be 2D batch, got {motion.shape}")

        if ref_idx < 0:
            ref_idx = len(motion) + ref_idx

        if reset_height:
            trans_reset_axis = [0, 2]
        else:
            trans_reset_axis = [0, 1, 2]

        T = motion.shape[0]

        # 提提取并处理pose
        pose = r6d_to_rotation_matrix(
            motion[:, self.pose_mask.to(motion.device)]
        ).reshape(T, self.n_joint, 3, 3)

        ref_motion = ref_motion.to(motion.device)
        # decode出来的pose固定是local pose
        ref_pose, ref_joint, _ = self.decode(ref_motion)

        # 把motion的骨架与ref的骨架进行同步
        if sync_skeleton:
            # 这里用的ref_pose是decode的pose 默认为local 所以global_pose=False
            ref_skel_offsets = self.body_model.get_skeleton_offsets(
                pose=ref_pose, joint=ref_joint, global_pose=False
            )
            if self.global_pose:
                fk = self.body_model.joint_fk_global
            else:
                fk = self.body_model.joint_fk_local
            joint_sync = fk(R=pose.clone(), skeleton_offsets=ref_skel_offsets)
            motion[:, self.joint_mask.to(motion.device)] = joint_sync.reshape(
                -1, self.n_joint * 3
            )
        # import pdb

        # pdb.set_trace()

        # global joint x-z初始值清零 朝向对齐
        R_ego_gv_inv = get_ego_gv(pose[ref_idx, 0]).transpose(-2, -1)
        R_ego_gv_ref = get_ego_gv(ref_pose[0, 0])
        ori_transform = R_ego_gv_ref.matmul(R_ego_gv_inv)

        if not self.global_pose:
            pose[:, 0] = ori_transform.matmul(pose[:, 0])
        else:
            pose = ori_transform.matmul(pose)

        pose = rotation_matrix_to_r6d(pose).reshape(-1, self.n_joint * 6)

        joint = (
            motion[:, self.joint_mask.to(motion.device)]
            .reshape(-1, self.n_joint, 3)
            .clone()
        )
        trans = motion[:, self.trans_mask.to(motion.device)].reshape(-1, 1, 3).clone()
        trans[:, :, [0, 2]] -= trans[:1, :, [0, 2]].clone()

        global_joint = joint + trans
        global_joint = ori_transform.matmul(global_joint.unsqueeze(-1)).squeeze(-1)

        # 方向已对齐 初始xz位移置0

        global_joint_ref = (
            self.get_component(ref_motion, "joint").reshape(-1, self.n_joint, 3).clone()
            + self.get_component(ref_motion, "trans").reshape(-1, 1, 3).clone()
        )

        # import pdb

        # pdb.set_trace()

        if stitch_joint_idx is None:
            stitch_joint_idx = torch.argmin(
                global_joint[0, :, 1]
            )  # 取ref的最低点作为对齐关节

        global_joint += (
            global_joint_ref[:1, [stitch_joint_idx]]
            - global_joint[:1, [stitch_joint_idx]]
        ).clone()

        if reset_height:
            # pass
            init_height = global_joint[0, 0, 1] - global_joint[0, :, 1].min()
            ground_ref = ref_joint[
                :60, :, 1
            ].min()  # 假设ref的最低点触地 其值视为地面高度
            global_joint[:, :, 1] -= global_joint[
                :1, [0], 1
            ].clone()  # motion初始高度置0
            global_joint[:, :, 1] += init_height + ground_ref  # 地面为ref时的高度

        trans = global_joint[:, 0].clone()
        joint = global_joint - global_joint[:, [0]]

        # 覆写
        motion[:, self.pose_mask] = pose.flatten(1)
        motion[:, self.joint_mask] = joint.flatten(1)
        motion[:, self.trans_mask] = trans.flatten(1)

        return motion

    def get_component(self, motion, component_name: str):
        assert component_name in ["pose", "joint", "trans"]
        return motion[..., self.mask_dict[component_name].to(motion.device)]

    @torch.no_grad()
    def motion_degradation_batch(
        self,
        motion: torch.Tensor,
        keyframe_mask=None,
        length=None,
        bool_length_mask=None,
    ):
        b = motion.shape[0]
        seq_len = motion.shape[1]
        motion = motion.clone()
        motion_origin = motion.clone()

        # 提提取并处理pose
        pose = r6d_to_rotation_matrix(
            motion[..., self.pose_mask.to(motion.device)]
        ).reshape(-1, self.n_joint, 3, 3)

        pose = pose.reshape(b, -1, self.n_joint, 3, 3)

        # 提提取并处理joint vel trans
        joint = motion[..., self.joint_mask.to(motion.device)].reshape(
            b, -1, self.n_joint, 3
        )

        trans = motion[..., self.trans_mask.to(motion.device)].reshape(b, -1, 3)

        pose, joint, trans = self.degradation.apply_random_degradations(
            pose, joint, trans, global_pose=self.global_pose
        )

        pose = rotation_matrix_to_r6d(pose).reshape(b, seq_len, -1)

        motion[..., self.pose_mask] = pose.flatten(2)
        motion[..., self.joint_mask] = joint.flatten(2)
        motion[..., self.trans_mask] = trans.flatten(2)

        if keyframe_mask is not None:
            keyframe_mask_bool = keyframe_mask == 1
            motion[keyframe_mask_bool] = motion_origin[keyframe_mask_bool]

        if bool_length_mask is not None:
            motion[~bool_length_mask] *= 0.0

        return motion

    def kinematic_loss_batch(
        self,
        R6d,
        joint,
        length=None,
        l1_weight=0.0,
        l2_weight=1.0,
    ):
        b = R6d.shape[0]
        n_j = self.n_joint

        R = r6d_to_rotation_matrix(R6d.clone()).reshape(-1, n_j, 3, 3)
        joint = joint.clone().reshape(b, -1, n_j, 3)

        offsets_from_motion = self.body_model.get_skeleton_offsets(
            pose=R, joint=joint, global_pose=self.global_pose, require_grad=True
        ).reshape(b, -1, n_j, 3)
        init_skeleton_offsets = offsets_from_motion[:, [0]]
        loss_rigid_body = (
            torch.nn.functional.mse_loss(
                offsets_from_motion,
                init_skeleton_offsets.expand_as(offsets_from_motion),
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
        if length is not None:
            mask = torch.zeros_like(loss_rigid_body[:, :, :1, :1])
            for i in range(b):
                mask[i, : length[i]] = 1.0
            mask_rigid_body = mask.expand_as(loss_rigid_body)
            loss_rigid_body = (
                loss_rigid_body * mask_rigid_body
            ).sum() / mask_rigid_body.sum()

        else:
            loss_rigid_body = loss_rigid_body.mean()

        return loss_rigid_body


def r6d_norm(r6d):
    origin_shape = r6d.shape
    r6d = r6d.reshape(-1, 6)
    column0 = normalize_tensor(r6d[:, 0:3])
    column1 = normalize_tensor(
        r6d[:, 3:6] - (column0 * r6d[:, 3:6]).sum(dim=1, keepdim=True) * column0
    )
    r = torch.cat([column0, column1], dim=-1)
    r = r.reshape(origin_shape)
    return r


def normalize_tensor(x: torch.Tensor, dim=-1, return_norm=False):
    """
    Normalize a tensor in a specific dimension to unit norm. (torch)

    :param x: Tensor in any shape.
    :param dim: The dimension to be normalized.
    :param return_norm: If True, norm(length) tensor will also be returned.
    :return: Tensor in the same shape. If return_norm is True, norm tensor in shape [*, 1, *] (1 at dim)
             will also be returned (keepdim=True).
    """
    norm = x.norm(dim=dim, keepdim=True)
    normalized_x = x / (norm)
    return normalized_x if not return_norm else (normalized_x, norm)


def get_temporal_mask(motion, length, mode="random_frame", loop_k=None, dtype="float"):
    assert dtype in ["float", "bool"]
    batch_size, n_frames, d_motion = motion.shape
    temporal_mask = torch.zeros(
        (batch_size, n_frames, d_motion), dtype=torch.float32, device=motion.device
    )
    lengths = length.reshape(-1)
    if mode == "uncond":
        pass
    elif mode == "random_frame":
        # Observe frames every trans_length frames
        # used for inference
        obs_rate = random.uniform(0.05, 0.1)  # 观察帧比例
        for i, length in enumerate(lengths.cpu().numpy()):
            length = int(length)
            obs_indices = random_index(data_len=length, sampling_rate=obs_rate)
            # 时间维度现在是第二个维度，调整索引位置
            temporal_mask[i, obs_indices] = 1  # set keyframes

    elif mode == "random_phrase":
        # Observe frames in random phrases
        # used for inference
        for i, length in enumerate(lengths.cpu().numpy()):
            length = int(length)
            length_phrases = [
                min(1, random.randint(1, length // 4)),
                min(1, random.randint(1, length // 8)),
            ]
            for length_phrase in length_phrases:  # 2次随机长度
                n_phrase = length // length_phrase
                phrase_idx_dict = {}
                for j in range(n_phrase):
                    phrase_idx_dict.update(
                        {
                            j: list(
                                range(
                                    j * length_phrase,
                                    min((j + 1) * length_phrase, length),
                                )
                            )
                        }
                    )
                # 注意：原代码中random_index未定义，这里保持原样
                ramdom_phrase_idx = random_index(
                    data_len=n_phrase,
                    sampling_rate=random.uniform(min(0.05, 1 / n_phrase), 0.25),
                )

                for j in ramdom_phrase_idx:
                    # 时间维度现在是第二个维度，调整索引位置
                    temporal_mask[i, phrase_idx_dict[j]] = 1  # set keyframes

    elif mode == "random_start_end":
        # 保障起始1到n帧和末尾部分mask=1，中间随机连续50%-90% mask=0
        for i, length in enumerate(lengths.cpu().numpy()):
            length = int(length)

            # 确定起始部分的长度n（1到总长度的1/10之间）
            max_start_length = max(1, length // 5)  # 起始部分最长不超过总长度的1/5
            start_length = random.randint(1, max_start_length)

            remaining_length = length - start_length
            # 计算中间mask=0区域的长度（剩余长度的50%-90%）
            min_mask0_length = int(remaining_length * 0.5)
            max_mask0_length = int(remaining_length * 0.9)
            mask0_length = random.randint(min_mask0_length, max_mask0_length)

            # 计算末尾部分的长度
            end_length = length - start_length - mask0_length
            end_length = max(end_length, 1)  # 确保末尾部分至少有1帧

            # 设置起始部分mask=1，调整索引位置
            temporal_mask[i, :start_length] = 1

            # 设置末尾部分mask=1，调整索引位置
            temporal_mask[i, -end_length:] = 1
    elif mode == "fix_start_end":
        assert loop_k is not None
        # 设置起始部分mask=1，调整索引位置
        temporal_mask[:, :loop_k] = 1
        # 设置末尾部分mask=1，调整索引位置
        temporal_mask[:, -loop_k:] = 1
    if dtype == "bool":
        temporal_mask = temporal_mask >= 1
    return temporal_mask


def random_index(data_len: int, sampling_rate=1.0, seed: int = None) -> list:
    """
    随机采样索引的函数

    参数:
    sample_size (int): 样本数量
    sampling_rate (float): 采样率，范围在(0, 1]
    seed (int or None): 随机数生成的种子，如果为None则不设置种子

    返回:
    list: 随机采样的索引列表
    """
    if not (0 < sampling_rate <= 1):
        raise ValueError("采样率必须在(0, 1]范围内")

    # 设置随机数种子
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(int(time.time()))

    # 计算需要采样的样本数量
    num_samples_to_select = max(int(data_len * sampling_rate), 1)

    # 生成样本索引的随机排列
    all_indices = np.arange(data_len)
    np.random.shuffle(all_indices)

    # 从随机排列的索引中选择需要的数量
    selected_indices = all_indices[:num_samples_to_select]

    return selected_indices
