import torch
import numpy as np
from .utils import quat_seq_resample, vec_seq_resample, get_ego_gv
from .motion_representation import HM263XRep, Motion291Rep
import os
from MotionLab.smplx_transform import smplx2body_motion
import MotionLab.articulate as art
from MotionLab.articulate.math.angular import (
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
    quaternion_to_rotation_matrix,
    axis_angle_to_rotation_matrix,
    rotation_matrix_to_r6d,
    r6d_to_rotation_matrix,
    rotation_matrix_to_axis_angle,
)
from tqdm import tqdm
import glob
from Aplus.tools.smpl_light import SMPLight
import copy


def avg_slide(pos: torch.Tensor, threshold: float = 0.015) -> torch.Tensor:
    """
    纯张量向量化计算平均滑步距离（修复索引越界，无显式循环）
    参数:
        pos: [T, n, 3] 时间步×关节数×三维坐标
        threshold: 触地判断阈值
    返回:
        avg_slide_dist: [n] 每个关节的平均滑步距离
    """
    T, n, _ = pos.shape
    device = pos.device
    dtype = pos.dtype

    # 步骤1：触地掩码 [T, n]，并过滤无效索引（防御性编程）
    pos[..., 1] -= pos[..., 1].min()
    contact_mask = pos[..., 1] < threshold  # T×n

    delta_pos = torch.norm(pos[1:, :, [0, 2]] - pos[:-1, :, [0, 2]], dim=-1)  # [T-1, n]
    contact_mask = contact_mask[:-1, :] & contact_mask[1:, :]  # [T-1, n]
    contact_mask = contact_mask.float()
    contact_times = contact_mask.sum(dim=0) + 1

    # import pdb
    # pdb.set_trace()

    return (delta_pos * contact_mask).sum(dim=0) / contact_times

    # contact_mask = contact_mask & (torch.arange(T, device=device)[:, None] < T)  # 时间索引≤T-1
    # contact_mask = contact_mask & (torch.arange(n, device=device)[None, :] < n)  # 关节索引≤n-1

    # # 步骤2：提取所有触地时刻的「时间索引」和「关节索引」（注意nonzero返回顺序：(行,列)=(时间,关节)）
    # touch_times, touch_joints = torch.nonzero(contact_mask, as_tuple=True)  # 均为[N,]，N是总触地次数
    # N = len(touch_times)

    # # 边界1：无任何触地时，直接返回全0
    # if N == 0:
    #     return torch.zeros(n, device=device, dtype=dtype)

    # # 步骤3：按关节分组，对每个关节的触地时间排序（核心修复：严格按关节分组处理）
    # # 3.1 为每个关节分配触地时刻的掩码
    # joint_masks = [touch_joints == j for j in range(n)]
    # # 3.2 预分配结果张量
    # avg_slide_dist = torch.zeros(n, device=device, dtype=dtype)

    # # 步骤4：并行化处理所有关节（通过向量化掩码，无显式循环）
    # # 方式：利用torch.where和掩码批量计算，避免单关节循环
    # for j in range(n):
    #     mask_j = joint_masks[j]
    #     times_j = touch_times[mask_j]  # 当前关节j的所有触地时间索引
    #     count_j = len(times_j)

    #     # 边界2：触地次数<2，跳过
    #     if count_j < 2:
    #         continue

    #     # 3.3 按时间排序当前关节的触地时刻（关键：确保时间递增）
    #     times_j_sorted, idx_sorted = torch.sort(times_j)
    #     # 3.4 提取排序后的触地坐标
    #     pos_j_touch = pos[times_j_sorted, j, :]  # [count_j, 3]
    #     # 3.5 计算相邻触地的位移和距离
    #     displacement_j = pos_j_touch[1:] - pos_j_touch[:-1]  # [count_j-1, 3]
    #     slide_dist_j = torch.norm(displacement_j, dim=-1)  # [count_j-1,]
    #     # 3.6 计算平均滑步距离
    #     avg_slide_dist[j] = slide_dist_j.mean()

    # return avg_slide_dist


class MotionProcessor:
    def __init__(
        self,
    ):
        self.pre_transform_R = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pose = []
        self.joint = []
        self.trans = []
        self.frame_rate = []

    def clear_cache(self):
        self.pose = []
        self.joint = []
        self.trans = []
        self.frame_rate = []

    @torch.no_grad()
    def load_and_normalize(self, path):
        pass

    def save(self, path):
        pass

    @torch.no_grad()
    def normalize(self, pose, joint, trans, vel=None):
        assert pose.shape[1] == 22, "Pose must have 22 joints (body only)"
        assert (
            joint.shape[1] == 22 or joint.shape[1] == 24
        ), "Joint positions must have 22/24 joints"
        device = self.device

        pose, joint, trans = pose.to(device), joint.to(device), trans.to(device)
        if vel is not None:
            vel = vel.to(device)

        root_oris = art.math.axis_angle_to_rotation_matrix(pose[:, 0])

        # y轴朝上
        if self.pre_transform_R is not None:
            R_SMPL = self.pre_transform_R.to(device)
            joint = R_SMPL.matmul(joint.unsqueeze(-1)).view_as(joint)
            trans = R_SMPL.matmul(trans.unsqueeze(-1)).view_as(trans)
            root_oris = R_SMPL.matmul(root_oris)

        # 人体面向世界坐标系z轴
        R_ego_gv_inv = get_ego_gv(root_oris[0]).transpose(-2, -1)
        joint = R_ego_gv_inv.matmul(joint.unsqueeze(-1)).view_as(joint)
        trans = R_ego_gv_inv.matmul(trans.unsqueeze(-1)).view_as(trans)
        if vel is not None:
            vel = R_ego_gv_inv.matmul(vel.unsqueeze(-1)).view_as(vel)
        root_oris = R_ego_gv_inv.matmul(root_oris)
        root_axis_angle = art.math.rotation_matrix_to_axis_angle(
            root_oris
        )  # 转回轴角格式再覆盖原始数值
        pose[:, 0] = root_axis_angle

        # 初始位置归位到x-z平面原点
        trans[:, [0, 2]] -= trans[:1, [0, 2]]
        joint[:, :, [0, 2]] -= joint[:1, :1, [0, 2]]

        # 初始触地
        init_h = joint[0, :, 1].min()
        trans[:, 1] -= init_h
        joint[:, :, 1] -= init_h

        result = [pose, joint, trans]

        if vel is not None:
            result.append(vel)

        return result

    def stationary_detect(self, positions, thres=2.5e-2):
        # import pdb; pdb.set_trace()
        # 处理numpy数组
        if isinstance(positions, np.ndarray) or isinstance(positions, torch.Tensor):
            velfactor = thres**2
        else:
            print("data type error")
            return None, None

        vel_3d = positions[1:] - positions[:-1]

        vel_joint_sq = (vel_3d**2).sum(dim=-1)  # (T-1, n_joint)

        stationary = vel_joint_sq < velfactor  # (T-1, n_joint)

        if isinstance(positions, np.ndarray):
            return np.array(stationary, dtype=np.float32)
        elif isinstance(positions, torch.Tensor):
            return stationary.float()
        else:
            return None

    def merge_data(self, mp):
        """
        合并另一个MotionProcessor
        """
        self.pose += mp.pose
        self.joint += mp.joint
        self.trans += mp.trans
        self.frame_rate += mp.frame_rate


class AMASSProcessor(MotionProcessor):
    def __init__(self, keep_hand=False):
        super().__init__()
        self.pre_transform_R = torch.FloatTensor(
            [[[1, 0, 0], [0, 0, 1], [0, -1, 0]]]
        ).to(self.device)
        self.keep_hand = keep_hand
        if keep_hand:
            self.n_joint = 24
        else:
            self.n_joint = 22

    @torch.no_grad()
    def load_and_normalize(self, path):
        amass_data = []
        for entry in os.listdir(path):
            entry_path = os.path.join(path, entry)
            if os.path.isdir(entry_path):
                amass_data.append(entry)

        for ds_name in amass_data:
            pose, joint, trans, frame_rate = [], [], [], []
            print("\rReading", ds_name)
            if ds_name in ["WEIZMANN", "humanact12"]:
                print("skip")
                continue
            npz_paths = glob.glob(os.path.join(path, ds_name, "*/*_poses.npz"))
            if len(npz_paths) == 0:
                npz_paths = glob.glob(os.path.join(path, ds_name, "*/*_stageii.npz"))
            for npz_fname in tqdm(npz_paths):
                try:
                    cdata = np.load(npz_fname, allow_pickle=True)
                except:
                    continue

                _pose, _joint, _trans, _frame_rate, beta = smplx2body_motion(
                    cdata, fps=None, keep_hand=self.keep_hand
                )
                _pose[:, 23] = _pose[:, 37]  # right hand
                if self.keep_hand:
                    _joint[:, 23] = _joint[:, 37]
                    _joint = _joint[:, :24]
                _pose = _pose[:, :24]  # body only

                # ===========Normalization===========
                _pose, _joint, _trans = self.normalize(_pose, _joint, _trans)

                pose.append(_pose.cpu())
                joint.append(_joint.cpu())
                trans.append(_trans.cpu())
                frame_rate.append(_frame_rate)
                # print(frame_rate)
            if len(pose) != 0:
                print(f"AMASS dataset {ds_name} loaded, {len(pose)} sequences found.")
            else:
                continue
            self.pose += pose
            self.joint += joint
            self.trans += trans
            self.frame_rate += frame_rate
            data_duration = 0
            for i in range(len(self.pose)):
                data_duration += float(len(self.pose[i])) / float(self.frame_rate[i])
            print(f"data data_duration: {data_duration / 3600} hours")

        # data_duration = 0
        # for i in range(len(self.pose)):
        #     data_duration += float(len(self.pose[i])) / float(self.frame_rate[i])
        # print(f'data data_duration: {data_duration/3600} hours')

    def save(self, path, fps=30, mirror=False):
        pose_out, joint_out, trans_out, joint_stationary_out = [], [], [], []
        print(f"Saving AMASS data to {path} at {fps} fps, mirror={mirror}...")
        target_fps = fps
        save_dir = path
        body_model = SMPLight()
        for i in tqdm(range(len(self.pose))):
            if len(self.pose[i]) < 30:
                continue
            _pose = self.pose[i]
            _joint = self.joint[i]
            _trans = self.trans[i]

            assert (
                self.n_joint == _joint.shape[1]
            ), f"Joint number mismatch: expected {self.n_joint}, got {_joint.shape[1]}"

            origin_fps = self.frame_rate[i]
            down_sample = origin_fps / target_fps

            # 0. 整数倍降采样逻辑（当原始fps是目标fps的整数倍时）
            if down_sample.is_integer() and down_sample >= 1.0:
                downsample_step = int(down_sample)
                _pose = _pose[::downsample_step]
                _joint = _joint[::downsample_step]
                _trans = _trans[::downsample_step]
            else:
                _pose = axis_angle_to_quaternion(_pose).reshape(
                    -1, 24, 4
                )  # 转换为四元数
                _pose = quat_seq_resample(
                    _pose,
                    original_fps=origin_fps,
                    target_fps=target_fps,
                    method="slerp",
                )
                _pose = quaternion_to_axis_angle(_pose).reshape(-1, 24, 3)
                _joint = vec_seq_resample(
                    _joint.flatten(1), original_fps=origin_fps, target_fps=target_fps
                ).reshape(-1, self.n_joint, 3)
                _trans = vec_seq_resample(
                    _trans, original_fps=origin_fps, target_fps=target_fps
                )

            if mirror:
                _pose = (
                    art.math.axis_angle_to_rotation_matrix(_pose)
                    .clone()
                    .view(-1, 24, 3, 3)
                )
                _pose = body_model.mirror_pose(_pose)
                _pose = art.math.rotation_matrix_to_axis_angle(_pose).view(-1, 24, 3)
                _joint = body_model.mirror_ric(_joint)
                _trans = body_model.mirror_3d_vecs(_trans)

            _joint_stationary = self.stationary_detect(_joint)
            pose_out.append(_pose[:-1])  # 鉴于速度计算会少一帧，pose和trans都少最后一帧
            joint_out.append(_joint[:-1])
            trans_out.append(_trans[:-1])
            joint_stationary_out.append(_joint_stationary)

        pose_out = torch.cat(pose_out, dim=0)
        joint_out = torch.cat(joint_out, dim=0)
        trans_out = torch.cat(trans_out, dim=0)
        joint_stationary_out = torch.cat(joint_stationary_out, dim=0)

        print("Saving")
        os.makedirs(path, exist_ok=True)
        if mirror:
            torch.save(pose_out, os.path.join(save_dir, "pose_m.pt"))
            torch.save(trans_out, os.path.join(save_dir, "tran_m.pt"))
            torch.save(joint_out, os.path.join(save_dir, "joint_m.pt"))
            torch.save(joint_stationary_out, os.path.join(save_dir, "stationary_m.pt"))
        else:
            torch.save(pose_out, os.path.join(save_dir, "pose.pt"))
            torch.save(trans_out, os.path.join(save_dir, "tran.pt"))
            torch.save(joint_out, os.path.join(save_dir, "joint.pt"))
            torch.save(joint_stationary_out, os.path.join(save_dir, "stationary.pt"))
        print(f"AMASS data is saved at", save_dir)


class Npz2VecProcessor(MotionProcessor):
    def __init__(self, keep_hand=False):
        super().__init__()
        self.pre_transform_R = None
        self.keep_hand = keep_hand
        if keep_hand:
            self.n_joint = 52
        else:
            self.n_joint = 22

    @torch.no_grad()
    def load_and_normalize(self, path):
        print(f"keep_hand={self.keep_hand}")
        motion_data = []
        for entry in os.listdir(path):
            entry_path = os.path.join(path, entry)
            if os.path.isdir(entry_path):
                motion_data.append(entry)

        for ds_name in motion_data:
            pose, joint, trans, frame_rate = [], [], [], []
            print("\rReading", ds_name)
            npz_paths = glob.glob(os.path.join(path, ds_name, "*.npz"))
            assert len(npz_paths) > 0, f"No npz files found in {ds_name}"
            for npz_fname in tqdm(npz_paths):

                _pose, _joint, _trans, _frame_rate = self.load_npz(npz_fname)

                # # ===========Normalization===========
                # _pose, _joint, _trans = self.normalize(_pose, _joint, _trans)

                pose.append(_pose)
                joint.append(_joint)
                trans.append(_trans)
                frame_rate.append(_frame_rate)

            assert (
                len(pose) != 0
            ), "AMASS dataset not found. Check config.py or comment the function process_amass()"
            self.pose += pose
            self.joint += joint
            self.trans += trans
            self.frame_rate += frame_rate
            data_duration = 0
            for i in range(len(self.pose)):
                data_duration += float(len(self.pose[i])) / float(self.frame_rate[i])
            print(f"data data_duration: {data_duration / 3600:.2f} hours")
        # data_duration = 0
        # for i in range(len(self.pose)):
        #     data_duration += float(len(self.pose[i])) / float(self.frame_rate[i])
        # print(f'data data_duration: {data_duration/3600} hours')

    @torch.no_grad()
    def load_npz(self, path):
        try:
            cdata = np.load(path, allow_pickle=True)
        except:
            FileNotFoundError(path)
            return

        pose, joint, trans, frame_rate, beta = smplx2body_motion(
            cdata, fps=None, keep_hand=True
        )
        # pose[:, 23] = pose[:, 37]  # right hand
        # pose = pose[:, :22]  # body only
        # # pose_hand = pose[:, 22:52]
        # if self.keep_hand:
        #     # joint[:, 23] = joint[:, 37]
        #     joint = joint[:, :22]
        return pose.cpu(), joint.cpu(), trans.cpu(), frame_rate

    def save(self, path, fps=30, mirror=False, seq_len=196):
        # joint_segments = []  # 用于收集所有符合长度的joint分段
        print(f"Saving data to {path} at {fps} fps, mirror={mirror}...")
        target_fps = fps
        save_dir = path
        body_model = SMPLight()
        idx = 0  # 用于编号保存的分段
        os.makedirs(save_dir, exist_ok=True)
        for i in tqdm(range(len(self.joint))):
            _joint = self.joint[i]

            origin_fps = self.frame_rate[i]
            down_sample = origin_fps / target_fps

            # 0. 整数倍降采样逻辑（当原始fps是目标fps的整数倍时）
            if down_sample.is_integer() and down_sample >= 1.0:
                downsample_step = int(down_sample)
                _joint = _joint[::downsample_step]
            else:
                _joint = vec_seq_resample(
                    _joint.flatten(1), original_fps=origin_fps, target_fps=target_fps
                ).reshape(-1, self.n_joint, 3)

            if len(_joint) < seq_len:
                continue

            if mirror:
                _joint = body_model.mirror_ric(_joint)

            num_segments = len(_joint) // seq_len
            for s in range(num_segments):
                start = s * seq_len
                end = start + seq_len
                segment = _joint[start:end]  # 截取一段
                # joint_segments.append(segment)
                segment_np = segment.cpu().numpy()  # 若在GPU上，先移到CPU
                if mirror:
                    filename = f"M{idx:06d}.npy"  # 镜像数据加_m后缀
                else:
                    filename = f"{idx:06d}.npy"
                idx += 1
                np.save(os.path.join(save_dir, filename), segment_np)

        # # 4. 保存所有分段（按000001.npy格式命名）

        # for idx, segment in enumerate(joint_segments, 0):  # 编号从0开始
        #     print(f"Saving segment {idx + 1} | {len(joint_segments)}", end="\r")
        #     # 生成6位数字文件名（如000000、000001...）

        #     # 转换为numpy数组并保存（.npy格式）
        #     segment_np = segment.cpu().numpy()  # 若在GPU上，先移到CPU
        #     np.save(os.path.join(path, filename), segment_np)

        print(f"Total {idx + 1} joint segments saved to {save_dir}")

    def save_as_vec263(self, path, fps=20, mirror=False, seq_len=224):
        # joint_segments = []  # 用于收集所有符合长度的joint分段
        print(f"Saving data to {path} at {fps} fps, mirror={mirror}...")
        target_fps = fps
        save_dir = path
        body_model = SMPLight()
        idx = 0  # 用于编号保存的分段
        data_rep = HM263XRep(keep_hand=self.keep_hand)
        os.makedirs(save_dir, exist_ok=True)
        for i in tqdm(range(len(self.pose))):
            if len(self.pose[i]) < 120:
                continue
            _pose = self.pose[i]
            _joint = self.joint[i]
            _trans = self.trans[i]

            origin_fps = self.frame_rate[i]
            down_sample = origin_fps / target_fps

            # print(len(_pose))

            # 0. 整数倍降采样逻辑（当原始fps是目标fps的整数倍时）
            if down_sample.is_integer() and down_sample >= 1.0:
                downsample_step = int(down_sample)
                _pose = _pose[::downsample_step]
                _joint = _joint[::downsample_step]
                _trans = _trans[::downsample_step]
            else:
                _pose = axis_angle_to_quaternion(_pose).reshape(
                    -1, 24, 4
                )  # 转换为四元数
                _pose = quat_seq_resample(
                    _pose,
                    original_fps=origin_fps,
                    target_fps=target_fps,
                    method="slerp",
                )
                _pose = quaternion_to_axis_angle(_pose).reshape(-1, 24, 3)
                _joint = vec_seq_resample(
                    _joint.flatten(1), original_fps=origin_fps, target_fps=target_fps
                ).reshape(-1, self.n_joint, 3)
                _trans = vec_seq_resample(
                    _trans, original_fps=origin_fps, target_fps=target_fps
                )

            if mirror:
                _pose = (
                    art.math.axis_angle_to_rotation_matrix(_pose)
                    .clone()
                    .view(-1, 24, 3, 3)
                )
                _pose = body_model.mirror_pose(_pose)
                _pose = art.math.rotation_matrix_to_axis_angle(_pose).view(-1, 24, 3)
                _joint = body_model.mirror_ric(_joint)
                _trans = body_model.mirror_3d_vecs(_trans)
            # print('reduce:', len(_pose))
            # print('--------------------')

            # _vel = _joint[1:] - _joint[:-1]
            _stationary = self.stationary_detect(_joint)
            _pose = _pose[:-1]
            _joint = _joint[:-1]
            _trans = _trans[:-1]

            _pose = axis_angle_to_rotation_matrix(_pose)
            _pose = rotation_matrix_to_r6d(_pose).reshape(-1, 24, 6)
            # import pdb; pdb.set_trace()
            _hm263x_rep = data_rep.encode(_pose, _joint, _stationary)

            num_segments = len(_joint) // seq_len

            for s in range(num_segments):
                start = s * seq_len
                end = start + seq_len
                segment = _hm263x_rep[start:end]  # 截取一段
                # joint_segments.append(segment)
                segment_np = segment.cpu().numpy()  # 若在GPU上，先移到CPU
                if mirror:
                    filename = f"M{idx:06d}.npy"  # 镜像数据加_m后缀
                else:
                    filename = f"{idx:06d}.npy"
                idx += 1
                np.save(os.path.join(save_dir, filename), segment_np)

            # 若数据长度小于设置的分段长度，则直接保存
            if num_segments == 0:
                segment_np = _hm263x_rep.cpu().numpy()  # 若在GPU上，先移到CPU
                if mirror:
                    filename = f"M{idx:06d}.npy"  # 镜像数据加_m后缀
                else:
                    filename = f"{idx:06d}.npy"
                idx += 1
                np.save(os.path.join(save_dir, filename), segment_np)

        print(f"Total {idx + 1} joint segments saved to {save_dir}")

    def save_as_vec291(self, path, fps=30, mirror=False, seq_len=256, global_pose=True):
        # joint_segments = []  # 用于收集所有符合长度的joint分段
        print(f"Saving data to {path} at {fps} fps, mirror={mirror}...")
        target_fps = fps
        save_dir = path
        body_model = SMPLight()
        idx = 0  # 用于编号保存的分段
        data_rep = Motion291Rep(keep_hand=self.keep_hand, global_pose=global_pose)
        os.makedirs(save_dir, exist_ok=True)
        for i in tqdm(range(len(self.pose))):
            if len(self.pose[i]) < 120:
                continue
            _pose = self.pose[i]
            _joint = self.joint[i]
            _trans = self.trans[i]

            origin_fps = self.frame_rate[i]
            down_sample = origin_fps / target_fps

            # print(len(_pose))

            # 0. 整数倍降采样逻辑（当原始fps是目标fps的整数倍时）
            if down_sample.is_integer() and down_sample >= 1.0:
                downsample_step = int(down_sample)
                _pose = _pose[::downsample_step]
                _joint = _joint[::downsample_step]
                _trans = _trans[::downsample_step]
            else:
                _pose = axis_angle_to_quaternion(_pose).reshape(
                    -1, 24, 4
                )  # 转换为四元数
                _pose = quat_seq_resample(
                    _pose,
                    original_fps=origin_fps,
                    target_fps=target_fps,
                    method="slerp",
                )
                _pose = quaternion_to_axis_angle(_pose).reshape(-1, 24, 3)
                _joint = vec_seq_resample(
                    _joint.flatten(1), original_fps=origin_fps, target_fps=target_fps
                ).reshape(-1, self.n_joint, 3)
                _trans = vec_seq_resample(
                    _trans, original_fps=origin_fps, target_fps=target_fps
                )

            if mirror:
                _pose = (
                    art.math.axis_angle_to_rotation_matrix(_pose)
                    .clone()
                    .view(-1, 24, 3, 3)
                )
                _pose = body_model.mirror_pose(_pose)
                _pose = art.math.rotation_matrix_to_axis_angle(_pose).view(-1, 24, 3)
                _joint = body_model.mirror_ric(_joint)
                _trans = body_model.mirror_3d_vecs(_trans)
            # print('reduce:', len(_pose))
            # print('--------------------')

            _vel = _joint[1:] - _joint[:-1]
            # _stationary = self.stationary_detect(_joint)
            _pose = _pose[:-1]
            _joint = _joint[:-1]
            _trans = _trans[:-1]

            _pose = axis_angle_to_rotation_matrix(_pose).reshape(-1, 24, 3, 3)
            # import pdb; pdb.set_trace()
            _hm263x_rep = data_rep.encode(_pose, _joint, _vel)

            num_segments = len(_joint) // seq_len

            for s in range(num_segments):
                start = s * seq_len
                end = start + seq_len
                segment = _hm263x_rep[start:end]  # 截取一段
                # joint_segments.append(segment)
                segment = segment.cpu()  # 若在GPU上，先移到CPU
                if mirror:
                    filename = f"M{idx:06d}.pt"  # 镜像数据加_m后缀
                else:
                    filename = f"{idx:06d}.pt"
                idx += 1
                torch.save(segment, os.path.join(save_dir, filename))

            # 若数据长度小于设置的分段长度，则直接保存
            if num_segments == 0:
                segment = _hm263x_rep.cpu()  # 若在GPU上，先移到CPU
                if mirror:
                    filename = f"M{idx:06d}.pt"  # 镜像数据加_m后缀
                else:
                    filename = f"{idx:06d}.pt"
                idx += 1
                torch.save(segment, os.path.join(save_dir, filename))

        print(f"Total {idx} joint segments saved to {save_dir}")

    def load_and_save_as_vec292(
        self,
        path_load,
        path_save,
        fps=30,
        mirror=False,
        seq_len=256,
        global_pose=True,
        max_batch=1000,
    ):
        # joint_segments = []  # 用于收集所有符合长度的joint分段
        print(f"Saving data to {path_save} at {fps} fps, mirror={mirror}...")
        target_fps = fps
        save_dir = path_save
        body_model = SMPLight()
        idx = 0  # 用于编号保存的分段
        data_rep = Motion291Rep(keep_hand=self.keep_hand, global_pose=global_pose)
        os.makedirs(save_dir, exist_ok=True)
        if mirror:
            mirror_set = [True, False]
        else:
            mirror_set = [False]

        motion_data = []
        for entry in os.listdir(path_load):
            entry_path = os.path.join(path_load, entry)
            if os.path.isdir(entry_path):
                motion_data.append(entry)

        for ds_name in motion_data:
            print("\rReading", ds_name)
            all_pass = False
            # if ds_name in ["ACCAD"]:
            #     all_pass = True
            # continue
            # if ds_name.lower().find("ssm_synced") != -1:
            #     print(f"skip {ds_name}")
            #     continue
            npz_paths = glob.glob(os.path.join(path_load, ds_name, "*.npz"))
            assert len(npz_paths) > 0, f"No npz files found in {ds_name}"
            # 按照batch分块处理，避免内存占用过高
            batch_num = (len(npz_paths) + max_batch - 1) // max_batch
            for b in range(batch_num):
                start_idx = b * max_batch
                end_idx = min((b + 1) * max_batch, len(npz_paths))
                npz_paths_batch = npz_paths[start_idx:end_idx]
                self.clear_cache()
                for npz_fname in tqdm(npz_paths_batch):

                    _pose, _joint, _trans, _frame_rate = self.load_npz(npz_fname)

                    # 测试jitter
                    jitter = (
                        (
                            (
                                _joint[3:]
                                - 3 * _joint[2:-1]
                                + 3 * _joint[1:-2]
                                - _joint[:-3]
                            )
                            * (fps * 1**3)
                        )
                        .norm(dim=2)
                        .mean()
                    )
                    if jitter > 1.0:
                        print(f"skip high jitter data: {npz_fname}, jitter: {jitter}")
                        npz_files = dict(np.load(npz_fname, allow_pickle=True))
                        np.savez(
                            os.path.join(
                                "FliterMotion/high_jitter_cases",
                                f"{npz_fname.split('/')[-2]}_{npz_fname.split('/')[-1]}",
                            ),
                            **npz_files,
                        )
                        continue

                    contact_joint = _joint[:, [4, 5, 7, 8, 10, 11, 20, 21]]
                    max_slide = avg_slide(contact_joint).max().item() * _frame_rate
                    # print(f"{npz_fname}, max_slide: {max_slide:.4f}")
                    if max_slide > 0.1:
                        print(
                            f"skip high slide data: {npz_fname} max_slide: {max_slide:.4f}"
                        )
                        npz_files = dict(np.load(npz_fname, allow_pickle=True))
                        np.savez(
                            os.path.join(
                                "FliterMotion/sliding_cases",
                                f"{npz_fname.split('/')[-2]}_{npz_fname.split('/')[-1]}",
                            ),
                            **npz_files,
                        )
                        # import pdb; pdb.set_trace()
                        continue

                    self.pose.append(_pose)
                    self.joint.append(_joint)
                    self.trans.append(_trans)
                    self.frame_rate.append(_frame_rate)

                if len(self.pose) != 0:
                    print(
                        f"AMASS dataset {ds_name} loaded, {len(self.pose)} sequences found."
                    )
                    # continue

                data_duration = 0
                for i in range(len(self.pose)):
                    data_duration += float(len(self.pose[i])) / float(
                        self.frame_rate[i]
                    )
                print(f"data_duration: {data_duration / 3600:.2f} hours")

                print(f"saving {ds_name}-seg_{b} mirror={mirror_set}")
                for i in tqdm(range(len(self.pose))):
                    if len(self.pose[i]) < 120:
                        continue
                    for mirror in mirror_set:
                        _pose = self.pose[i].to(self.device)
                        _joint = self.joint[i].to(self.device)
                        _trans = self.trans[i].to(self.device)

                        origin_fps = self.frame_rate[i]
                        down_sample = origin_fps / target_fps
                        # 0. 整数倍降采样逻辑（当原始fps是目标fps的整数倍时）
                        if down_sample.is_integer() and down_sample >= 1.0:
                            downsample_step = int(down_sample)
                            _pose = _pose[::downsample_step]
                            _joint = _joint[::downsample_step]
                            _trans = _trans[::downsample_step]
                        else:
                            _pose = axis_angle_to_quaternion(_pose).reshape(
                                -1, 24, 4
                            )  # 转换为四元数
                            _pose = quat_seq_resample(
                                _pose,
                                original_fps=origin_fps,
                                target_fps=target_fps,
                                method="slerp",
                            )
                            _pose = quaternion_to_axis_angle(_pose).reshape(-1, 24, 3)
                            _joint = vec_seq_resample(
                                _joint.flatten(1),
                                original_fps=origin_fps,
                                target_fps=target_fps,
                            ).reshape(-1, self.n_joint, 3)
                            _trans = vec_seq_resample(
                                _trans, original_fps=origin_fps, target_fps=target_fps
                            )

                        if mirror:
                            _pose = (
                                art.math.axis_angle_to_rotation_matrix(_pose)
                                .clone()
                                .view(-1, 24, 3, 3)
                            )
                            _pose = body_model.mirror_pose(_pose)
                            _pose = art.math.rotation_matrix_to_axis_angle(_pose).view(
                                -1, 24, 3
                            )
                            _joint = body_model.mirror_ric(_joint)
                            _trans = body_model.mirror_3d_vecs(_trans)

                        _vel = _joint[1:] - _joint[:-1]
                        _pose = _pose[:-1]
                        _joint = _joint[:-1]
                        _trans = _trans[:-1]

                        _pose = axis_angle_to_rotation_matrix(_pose).reshape(
                            -1, 24, 3, 3
                        )
                        _hm263x_rep = data_rep.encode(_pose, _joint, _vel)

                        num_segments = len(_joint) // seq_len

                        for s in range(num_segments):
                            start = s * seq_len
                            end = start + seq_len
                            segment = _hm263x_rep[start:end].cpu()  # 截取一段
                            if mirror:
                                filename = f"M{idx:06d}.pt"  # 镜像数据加_m后缀
                            else:
                                filename = f"{idx:06d}.pt"

                        # 若数据长度小于设置的分段长度，则直接保存
                        if num_segments == 0:
                            segment = _hm263x_rep.cpu()  # 若在GPU上，先移到CPU
                            if mirror:
                                filename = f"M{idx:06d}.pt"  # 镜像数据加_m后缀
                            else:
                                filename = f"{idx:06d}.pt"
                        idx += 1
                        torch.save(segment, os.path.join(save_dir, filename))

        print(f"Total {idx} joint segments saved to {save_dir}")

    def load_and_save(
        self,
        path_load,
        path_save,
        fps=30,
        mirror=False,
        seq_len=256,
        global_pose=True,
        max_batch=1000,
    ):
        # joint_segments = []  # 用于收集所有符合长度的joint分段
        print(f"Saving data to {path_save} at {fps} fps, mirror={mirror}...")
        target_fps = fps
        save_dir = path_save
        # body_model = SMPLight()
        # data_rep = Motion291Rep(keep_hand=self.keep_hand, global_pose=global_pose)
        os.makedirs(save_dir, exist_ok=True)
        # if mirror:
        # mirror_set = [True, False]
        # else:
        # mirror_set = [False]

        mirror_set = [False]

        motion_data = []
        for entry in os.listdir(path_load):
            entry_path = os.path.join(path_load, entry)
            if os.path.isdir(entry_path):
                motion_data.append(entry)

        for ds_name in motion_data:
            idx = 0  # 用于编号保存的分段
            print("\rReading", ds_name)
            all_pass = False
            # if ds_name in ["ACCAD"]:
            #     all_pass = True
            # continue
            # if ds_name.lower().find("ssm_synced") != -1:
            #     print(f"skip {ds_name}")
            #     continue
            npz_paths = glob.glob(os.path.join(path_load, ds_name, "*.npz"))
            assert len(npz_paths) > 0, f"No npz files found in {ds_name}"
            # 按照batch分块处理，避免内存占用过高
            batch_num = (len(npz_paths) + max_batch - 1) // max_batch
            for b in range(batch_num):
                start_idx = b * max_batch
                end_idx = min((b + 1) * max_batch, len(npz_paths))
                npz_paths_batch = npz_paths[start_idx:end_idx]
                self.clear_cache()
                for npz_fname in tqdm(npz_paths_batch):

                    _pose, _joint, _trans, _frame_rate = self.load_npz(npz_fname)

                    # 测试jitter
                    jitter = (
                        (
                            (
                                _joint[3:]
                                - 3 * _joint[2:-1]
                                + 3 * _joint[1:-2]
                                - _joint[:-3]
                            )
                            * (fps * 1**3)
                        )[:, :22]
                        .norm(dim=2)
                        .mean()
                    )
                    if jitter > 1.0:
                        print(f"skip high jitter data: {npz_fname}, jitter: {jitter}")
                        npz_files = dict(np.load(npz_fname, allow_pickle=True))
                        np.savez(
                            os.path.join(
                                "FliterMotion/high_jitter_cases",
                                f"{npz_fname.split('/')[-2]}_{npz_fname.split('/')[-1]}",
                            ),
                            **npz_files,
                        )
                        continue

                    # contact_joint = _joint[:, [4, 5, 7, 8, 10, 11, 20, 21]]
                    contact_joint = _joint[:, [10, 11]]
                    max_slide = avg_slide(contact_joint).min().item() * _frame_rate
                    # print(f"{npz_fname}, max_slide: {max_slide:.4f}")
                    if max_slide > 0.1:
                        print(
                            f"skip high slide data: {npz_fname} min_slide: {max_slide:.4f}"
                        )
                        npz_files = dict(np.load(npz_fname, allow_pickle=True))
                        np.savez(
                            os.path.join(
                                "FliterMotion/sliding_cases",
                                f"{npz_fname.split('/')[-2]}_{npz_fname.split('/')[-1]}",
                            ),
                            **npz_files,
                        )
                        # import pdb; pdb.set_trace()
                        continue

                    self.pose.append(_pose)
                    self.joint.append(_joint)
                    # self.trans.append(_trans)
                    self.frame_rate.append(_frame_rate)

                if len(self.pose) != 0:
                    print(
                        f"AMASS dataset {ds_name} loaded, {len(self.pose)} sequences found."
                    )
                    # continue

                data_duration = 0
                for i in range(len(self.pose)):
                    data_duration += float(len(self.pose[i])) / float(
                        self.frame_rate[i]
                    )
                print(f"data_duration: {data_duration / 3600:.2f} hours")

                print(f"saving {ds_name}-seg_{b} mirror={mirror_set}")
                for i in tqdm(range(len(self.pose))):
                    if len(self.pose[i] / self.frame_rate[i]) < 2.0:
                        continue
                    for mirror in mirror_set:
                        _pose = self.pose[i].to(self.device)
                        _joint = self.joint[i].to(self.device)
                        # 初始贴地
                        ground_h = _joint[:, :, 1].min()
                        _joint[:, :, 1] -= ground_h

                        _trans = _joint[:, 0].clone().to(self.device)
                        # 转换为相对root的joint position
                        _joint -= _trans.unsqueeze(1)

                        # import pdb; pdb.set_trace()

                        origin_fps = self.frame_rate[i]
                        down_sample = origin_fps / target_fps
                        # 0. 整数倍降采样逻辑（当原始fps是目标fps的整数倍时）
                        if down_sample.is_integer() and down_sample >= 1.0:
                            downsample_step = int(down_sample)
                            _pose = _pose[::downsample_step]
                            _joint = _joint[::downsample_step]
                            _trans = _trans[::downsample_step]
                        else:
                            _pose = axis_angle_to_quaternion(_pose).reshape(
                                -1, 52, 4
                            )  # 转换为四元数
                            _pose = quat_seq_resample(
                                _pose,
                                original_fps=origin_fps,
                                target_fps=target_fps,
                                method="slerp",
                            )
                            _pose = quaternion_to_axis_angle(_pose).reshape(
                                -1, self.n_joint, 3
                            )
                            _joint = vec_seq_resample(
                                _joint.flatten(1),
                                original_fps=origin_fps,
                                target_fps=target_fps,
                            ).reshape(-1, self.n_joint, 3)
                            _trans = vec_seq_resample(
                                _trans, original_fps=origin_fps, target_fps=target_fps
                            )

                        # if mirror:
                        #     _pose = (
                        #         art.math.axis_angle_to_rotation_matrix(_pose)
                        #         .clone()
                        #         .view(-1, 24, 3, 3)
                        #     )
                        #     _pose = body_model.mirror_pose(_pose)
                        #     _pose = art.math.rotation_matrix_to_axis_angle(_pose).view(-1, 24, 3)
                        #     _joint = body_model.mirror_ric(_joint)
                        #     _trans = body_model.mirror_3d_vecs(_trans)

                        _pose = axis_angle_to_rotation_matrix(_pose).reshape(
                            -1, self.n_joint, 3, 3
                        )

                        num_segments = len(_joint) // seq_len

                        for s in range(num_segments):
                            start = s * seq_len
                            end = start + seq_len
                            segment = {
                                "pose": _pose[start:end].cpu(),
                                "joint": _joint[start:end].cpu(),
                                "trans": _trans[start:end].cpu(),
                            }

                            if mirror:
                                filename = (
                                    f"{ds_name}_M{idx:06d}.pt"  # 镜像数据加_m后缀
                                )
                            else:
                                filename = f"{ds_name}_{idx:06d}.pt"

                        # 若数据长度小于设置的分段长度，则直接保存
                        if num_segments == 0:
                            segment = {
                                "pose": _pose.cpu(),
                                "joint": _joint.cpu(),
                                "trans": _trans.cpu(),
                            }  # 若在GPU上，先移到CPU
                            if mirror:
                                filename = (
                                    f"{ds_name}_M{idx:06d}.pt"  # 镜像数据加_m后缀
                                )
                            else:
                                filename = f"{ds_name}_{idx:06d}.pt"
                        idx += 1
                        torch.save(segment, os.path.join(save_dir, filename))

        print(f"Total {idx} joint segments saved to {save_dir}")

    def npz2motion(self, npz_fname, target_fps=20, chunk_size=1000):
        """
        分段处理SMPL数据以避免显存溢出
        Args:
            npz_fname: npz文件路径
            target_fps: 目标帧率
            chunk_size: 分段大小（每chunk处理的帧数，默认1000）
        """
        try:
            # 先加载整个文件获取基本信息
            cdata = np.load(npz_fname, allow_pickle=True)
        except:
            print(f"Failed to load {npz_fname}")
            return

        gender = cdata["gender"].item()
        betas = cdata["betas"].reshape(-1)[:16] if "betas" in cdata else np.zeros(10)
        if "mocap_framerate" in cdata.keys():
            mocap_framerate = int(cdata["mocap_framerate"])
        else:
            mocap_framerate = int(cdata["mocap_frame_rate"])
        
        # 获取poses和trans的总帧数
        poses_all = cdata["poses"]
        trans_all = cdata["trans"] if "trans" in cdata else np.zeros((poses_all.shape[0], 3))
        total_frames = poses_all.shape[0]
        
        # 确定分段数量
        num_chunks = (total_frames + chunk_size - 1) // chunk_size
        # print(f"Processing {npz_fname}: total_frames={total_frames}, chunks={num_chunks}, chunk_size={chunk_size}")
        
        # 存储各分段的处理结果
        pose_chunks = []
        joint_chunks = []
        trans_chunks = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_frames)
            
            # 创建当前chunk的数据字典
            chunk_data = {}
            chunk_data["gender"] = gender
            chunk_data["betas"] = betas
            chunk_data["mocap_framerate"] = mocap_framerate
            chunk_data["poses"] = poses_all[start_idx:end_idx]
            chunk_data["trans"] = trans_all[start_idx:end_idx]
            
            # 处理当前chunk
            _pose, _joint, _trans, _frame_rate, _ = smplx2body_motion(
                chunk_data, fps=None, keep_hand=self.keep_hand
            )
            
            # 重整形并截取到n_joint
            _pose = _pose.reshape(_pose.shape[0], -1, 3)
            _pose = _pose[:, :self.n_joint]
            _joint = _joint[:, :self.n_joint]
            
            pose_chunks.append(_pose)
            joint_chunks.append(_joint)
            trans_chunks.append(_trans)
            
            # print(f"  Chunk {chunk_idx+1}/{num_chunks}: frames {start_idx}-{end_idx} processed")
            
            # 清理当前chunk的内存
            del chunk_data, _pose, _joint, _trans
            
        # 合并所有chunk的结果
        _pose = torch.cat(pose_chunks, dim=0)
        _joint = torch.cat(joint_chunks, dim=0)
        _trans = torch.cat(trans_chunks, dim=0)
        _frame_rate = _frame_rate  # 最后一个chunk的frame_rate
        
        # 清理chunks列表的内存
        del pose_chunks, joint_chunks, trans_chunks
        
        origin_fps = _frame_rate
        down_sample = origin_fps / target_fps

        # 0. 整数倍降采样逻辑（当原始fps是目标fps的整数倍时）
        if down_sample.is_integer() and down_sample >= 1.0:
            downsample_step = int(down_sample)
            _pose = _pose[::downsample_step]
            _joint = _joint[::downsample_step]
            _trans = _trans[::downsample_step]
        else:
            _pose = axis_angle_to_quaternion(_pose).reshape(-1, self.n_joint, 4)  # 转换为四元数
            _pose = quat_seq_resample(
                _pose,
                original_fps=origin_fps,
                target_fps=target_fps,
                method="slerp",
            )
            _pose = quaternion_to_axis_angle(_pose).reshape(-1, self.n_joint, 3)
            _joint = vec_seq_resample(
                _joint.flatten(1), original_fps=origin_fps, target_fps=target_fps
            ).reshape(-1, self.n_joint, 3)
            _trans = vec_seq_resample(
                _trans, original_fps=origin_fps, target_fps=target_fps
            )

        return _pose.cpu(), _joint.cpu(), _trans.cpu(), betas, gender

    def motion2npz_dict(self, pose, trans, frame_rate, betas=None, gender=None):
        smplx_data = {}
        T, J = pose.shape[0], pose.shape[1]
        if pose.shape[-1] == 6:
            pose = r6d_to_rotation_matrix(pose)
        pose = rotation_matrix_to_axis_angle(pose).reshape(T, J, 3)

        pose_52 = torch.zeros(pose.shape[0], 52, 3).to(pose.device)
        # import pdb; pdb.set_trace()
        pose_52[:, :J] = pose[:, :J]

        if betas is None:
            betas = np.zeros(10)

        smplx_data["poses"] = np.array(pose_52.detach().cpu(), dtype=np.float32)
        smplx_data["trans"] = np.array(trans.detach().cpu())
        smplx_data["mocap_framerate"] = float(frame_rate)
        smplx_data["num_frames"] = int(len(pose))
        smplx_data["betas"] = betas
        # smplx_data["exp"] = np.zeros((len(pose), 50), dtype=np.float32)
        if gender is None:
            gender = "neutral"
        smplx_data["gender"] = gender
        # import pdb; pdb.set_trace()

        return smplx_data
