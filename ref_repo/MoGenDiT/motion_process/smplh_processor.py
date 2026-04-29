import torch
import numpy as np
from .utils import quat_seq_resample, vec_seq_resample, get_ego_gv
import os
import articulate as art
from articulate.math.angular import (
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
    quaternion_to_rotation_matrix,
    axis_angle_to_rotation_matrix,
    rotation_matrix_to_r6d,
    r6d_to_rotation_matrix,
    rotation_matrix_to_axis_angle
)
from tqdm import tqdm
from glob import glob
import copy

SMPL_JOINTS_FLIP_PERM = [
    0, 2, 1, 3, 5, 4, 6, 8, 7, 9,
    11, 10, 12, 14, 13, 15, 17, 16, 19, 18,
    21, 20,
]  # Hands are removed: # , 23, 22]
# fmt: on

male_bm_path = "./motion_process/body_model/smplh/male/model.npz"
male_dmpl_path = "./motion_process/body_model/dmpls/male/model.npz"

female_bm_path = "./motion_process/body_model/smplh/female/model.npz"
female_dmpl_path = "./motion_process/body_model/dmpls/female/model.npz"

neutral_bm_path = "./motion_process/body_model/smplh/neutral/model.npz"
neutral_dmpl_path = "./motion_process/body_model/dmpls/neutral/model.npz"

num_betas = 16  # number of body parameters
num_dmpls = 8  # number of DMPL parameters

comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from motion_process.human_body_prior.body_model.body_model import BodyModel

# 创建body model
male_bm = BodyModel(
    bm_fname=male_bm_path,
    num_betas=num_betas,
    num_dmpls=num_dmpls,
    dmpl_fname=male_dmpl_path,
).to(comp_device)

female_bm = BodyModel(
    bm_fname=female_bm_path,
    num_betas=num_betas,
    num_dmpls=num_dmpls,
    dmpl_fname=female_dmpl_path,
).to(comp_device)
neutral_bm = BodyModel(
    bm_fname=neutral_bm_path,
    num_betas=num_betas,
    num_dmpls=num_dmpls,
    dmpl_fname=neutral_dmpl_path,
).to(comp_device)

def loop_amams(
    base_folder,
    new_base_folder,
    ext=".npz",
    newext=".npz",
    force_redo=False,
    only_mirror=False,
    exclude=None,
):
    # 收集所有匹配的文件路径
    matches = []
    
    # 使用os.walk递归遍历目录
    for root, dirs, files in os.walk(base_folder):
        # 如果only_mirror为True，只处理M开头的目录
        if only_mirror:
            # 检查当前目录路径是否包含M目录
            rel_path = os.path.relpath(root, base_folder)
            if not rel_path.startswith("M_") and rel_path != ".":
                continue
        
        for file in files:
            if file.endswith(ext):
                # 获取相对于base_folder的相对路径
                motion_file = os.path.relpath(os.path.join(root, file), base_folder)
                matches.append(motion_file)
    
    # 按字母顺序排序以确保一致性
    matches.sort()
    
    for motion_file in tqdm(matches):
        if exclude and exclude in motion_file:
            continue

        motion_path = os.path.join(base_folder, motion_file)

        if motion_path.endswith("shape.npz"):
            continue

        # if motion_path.find("stagei") > -1 and motion_path.find("moyo") == -1:
        #     continue

        new_motion_path = os.path.join(
            new_base_folder, motion_file.replace(ext, newext)
        )
        if not force_redo and os.path.exists(new_motion_path):
            continue

        new_folder = os.path.split(new_motion_path)[0]
        os.makedirs(new_folder, exist_ok=True)

        yield motion_path, new_motion_path

def flip_pose(_pose):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """
    pose = _pose.clone()
    pose[..., :22, :] = pose[..., SMPL_JOINTS_FLIP_PERM, :]
    # we also negate the second and the third dimension of the axis-angle
    pose[...,:22, 1] = -pose[...,:22, 1]
    pose[...,:22, 2] = -pose[...,:22, 2]
    return pose

def flip_trans(_trans):
    # flip trans
    x, y, z = torch.unbind(_trans, dim=-1)
    mirrored_trans = torch.stack((-x, y, z), dim=-1)
    return mirrored_trans

@torch.no_grad()
def smplh_to_body_motion(smplh_data, beta=None, fps=None):
    """Convert SMPL-H motion data to body motion (joint rotation, joint position, trans)."""
    if "mocap_framerate" in smplh_data:
        mocap_framerate = int(smplh_data["mocap_framerate"])
    else:
        mocap_framerate = int(smplh_data["mocap_frame_rate"])
    pose, trans = (
        smplh_data["poses"],
        smplh_data["trans"],
    )
    assert pose.shape[1] == 52 * 3, f"Expected pose shape (N, 156), got {pose.shape}"

    beta = smplh_data["betas"].reshape(-1)[:16] if beta is None else beta

    pose = torch.FloatTensor(pose).flatten(1)
    trans = torch.FloatTensor(trans).flatten(1)

    if fps is not None:
        down_sample = mocap_framerate / fps
        # 0. 整数倍降采样逻辑（当原始fps是目标fps的整数倍时）
        if down_sample.is_integer() and down_sample >= 1.0:
            downsample_step = int(down_sample)
            pose = pose[::downsample_step]
            trans = trans[::downsample_step]
        else:
            pose = axis_angle_to_quaternion(pose).reshape(
                pose.shape[0], 52, 4
            )  # 转换为四元数
            pose = quat_seq_resample(
                pose,
                original_fps=mocap_framerate,
                target_fps=fps,
                method="slerp",
            )
            pose = quaternion_to_axis_angle(pose).reshape(pose.shape[0], -1)
            trans = vec_seq_resample(
                trans, original_fps=mocap_framerate, target_fps=fps
            )
        mocap_framerate = fps

    n_frames = pose.shape[0]

    if smplh_data["gender"] == "male":
        bm = male_bm
    elif smplh_data["gender"] == "female":
        bm = female_bm
    else:
        bm = neutral_bm

    body_parms = {
        "root_orient": pose[:, : 1 * 3].to(comp_device).detach(),
        "pose_body": pose[:, 1 * 3 : 22 * 3].to(comp_device).detach(),
        "pose_hand": pose[:, 22 * 3 : 52 * 3].to(comp_device).detach(),
        "trans": trans.to(comp_device).detach(),
        "betas": torch.Tensor(
            np.repeat(beta[:num_betas][np.newaxis], repeats=len(pose), axis=0)
        ).to(comp_device).detach(),
        "gender": smplh_data["gender"]
    }

    with torch.no_grad():
        body = bm(**body_parms)
    glob_jp = body.Jtr  # (N, 22, 3) global joint positions
    trans = glob_jp[:, 0].clone()  # (N, 3) global root joint position
    joint = glob_jp - trans.unsqueeze(1)  # (N, 22, 3) joint positions relative to root

    return {
        "poses": pose.reshape(n_frames, -1, 3).detach().cpu(),
        "joint": joint.reshape(n_frames, -1, 3).detach().cpu(),
        "trans": trans.detach().cpu(),
        "mocap_framerate": mocap_framerate,
        "betas": beta,
        "gender": smplh_data["gender"]
    }


def motion2npz_dict(pose, trans, frame_rate, betas=None, gender=None):
    smplh_data = {}
    n_j = pose.shape[1]
    if pose.shape[-1] == 6:
        pose = r6d_to_rotation_matrix(pose)
        pose = rotation_matrix_to_axis_angle(pose).reshape(-1, n_j, 3)
    elif len(pose.shape) ==4 and pose.shape[-1] == 3 and pose.shape[-2] == 3:
        pose = rotation_matrix_to_axis_angle(pose).reshape(-1, n_j, 3)
    elif len(pose.shape) == 3 and pose.shape[-1] == 3:
        pose = pose
    else:
        print(f'unsupport pose format {pose.shape}')
        return None

    pose_52 = torch.zeros(pose.shape[0], 52, 3)
    # import pdb; pdb.set_trace()
    pose_52[:, :n_j] = pose[:, :n_j]
    pose_52 = pose_52.reshape(-1, 52 * 3)

    if betas is None:
        betas = np.zeros(10)
    if gender is None:
        gender = 'neutral'

    smplh_data["poses"] = np.array(pose_52.detach().cpu(), dtype=np.float32)
    smplh_data["trans"] = np.array(trans.detach().cpu())
    smplh_data["mocap_framerate"] = float(frame_rate)
    smplh_data["num_frames"] = int(len(pose))
    smplh_data["betas"] = betas
    smplh_data["gender"] = gender

    return smplh_data

def mirror_dataset(base_folder, force_redo=False,):
    """
    镜像处理数据集：逐个读取根目录下的第一级文件夹（数据集名称），
    遍历其所有子文件夹中的npz motion数据并进行镜像处理，
    存储到根目录下的M_{数据集名称}文件夹中，保留原始文件夹目录结构。
    
    参数:
        base_folder: 根目录路径
        force_redo: 是否强制重新处理已存在的文件
        verify_integrity: 是否验证原始数据完整性
    """
    print("Mirroring dataset motion files")
    print(f"Base folder: {base_folder}")
    
    # 安全验证：确保不会意外修改原始数据
    if not os.path.exists(base_folder):
        raise ValueError(f"Base folder does not exist: {base_folder}")
    
    # 获取根目录下的第一级文件夹（数据集名称）
    dataset_names = []
    for entry in os.listdir(base_folder):
        entry_path = os.path.join(base_folder, entry)
        # 跳过非文件夹与镜像数据文件夹
        if os.path.isdir(entry_path) and not entry.startswith('M_'):
            dataset_names.append(entry)
    
    print(f"Found datasets: {dataset_names}")
    
    for dataset_name in dataset_names:
        print(f"\nProcessing dataset: {dataset_name}")
        if dataset_name not in ['LARa']:
            continue
        
        # 创建镜像文件夹
        mirror_dataset_name = f"M_{dataset_name}"
        new_base_folder = os.path.join(base_folder, mirror_dataset_name)
        
        print(f"Mirrored data will be stored in: {new_base_folder}")
        
        # 使用loop_amams遍历该数据集下的所有npz文件
        dataset_folder = os.path.join(base_folder, dataset_name)
        iterator = loop_amams(
            dataset_folder,
            new_base_folder,
            ext=".npz",
            newext=".npz",
            force_redo=force_redo,
            only_mirror=False,
        )
        
        processed_count = 0
        error_count = 0
        integrity_issues = 0
        skipped_count = 0
        
        for motion_path, new_motion_path in iterator:
            
            try:
                # 加载原始数据（创建深拷贝确保隔离）
                original_data = np.load(motion_path, allow_pickle=True)
                data = {}
                for key in original_data.keys():
                    data[key] = original_data[key].copy()         
                # 处理pose和trans数据，支持poses和pose键名兼容
                poses = torch.from_numpy(data["poses"]).reshape(len(data["poses"]), -1, 3)
                trans = torch.from_numpy(data["trans"])
                
                # 镜像处理
                mirror_poses = flip_pose(poses)
                mirrored_trans = flip_trans(trans)

                data["poses"] = np.array(mirror_poses.flatten(1))
                data["trans"] = np.array(mirrored_trans)
                    
                # 确保目标文件夹存在
                os.makedirs(os.path.dirname(new_motion_path), exist_ok=True)
                
                # 保存镜像数据
                np.savez(new_motion_path, **data)
                
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} files...")
                    
            except Exception as e:
                print(f"Error processing {motion_path}: {e}")
                error_count += 1
                continue
        
        print(f"Dataset {dataset_name}: Processed {processed_count} files, Errors: {error_count}, Integrity issues: {integrity_issues}, Skipped: {skipped_count}")
    
    print("\nMirroring completed!")
    if integrity_issues > 0:
        print(f"WARNING: Found {integrity_issues} potential integrity issues. Check logs for details.")
    else:
        print("All original files appear to be unchanged.")

def segment_dataset(base_folder, new_base_folder, max_segment_duration=16, force_redo=False):
    """
    对数据集中的npz动作文件进行多段切分，将超过指定时长的动作切分为多段
    
    参数:
        base_folder: 原始数据根目录路径
        new_base_folder: 切分后数据存储根目录路径
        max_segment_duration: 每段最大时长（秒），默认10秒
        force_redo: 是否强制重新处理已存在的文件
    """
    print("Segmenting dataset motion files")
    print(f"Base folder: {base_folder}")
    print(f"Max segment duration: {max_segment_duration} seconds")
    
    # 安全验证：确保不会意外修改原始数据
    if not os.path.exists(base_folder):
        raise ValueError(f"Base folder does not exist: {base_folder}")
    
    # 获取根目录下的第一级文件夹（数据集名称）
    dataset_names = []
    for entry in os.listdir(base_folder):
        entry_path = os.path.join(base_folder, entry)
        # 跳过非文件夹与已切分数据文件夹
        if os.path.isdir(entry_path) and not entry.startswith('Segmented_'):
            dataset_names.append(entry)
    
    print(f"Found datasets: {dataset_names}")
    
    for dataset_name in dataset_names:
        print(f"\nProcessing dataset: {dataset_name}")
        # if dataset_name.find('M_') ==0:
        #     print(f'skip_mirrored dataset {dataset_name}')
        #     continue
        # if dataset_name not in ['CNRS', 'GRAB', 'SOMA']:
        #     continue
        
        # 创建切分文件夹
        segmented_dataset_name = f"Segmented_{dataset_name}"
        new_dataset_folder = os.path.join(new_base_folder, segmented_dataset_name)
        
        print(f"Segmented data will be stored in: {new_dataset_folder}")
        
        # 使用loop_amams遍历该数据集下的所有npz文件
        dataset_folder = os.path.join(base_folder, dataset_name)
        iterator = loop_amams(
            dataset_folder,
            new_dataset_folder,
            ext=".npz",
            newext=".npz",
            force_redo=force_redo,
            only_mirror=False,
        )
        
        processed_count = 0
        error_count = 0
        segmented_count = 0
        skipped_count = 0
        
        for motion_path, new_motion_path in iterator:
            
            try:
                # 加载原始数据
                original_data = np.load(motion_path, allow_pickle=True)
                
                # 获取帧率信息
                if "mocap_framerate" in original_data:
                    frame_rate = float(original_data["mocap_framerate"])
                elif "mocap_frame_rate" in original_data:
                    frame_rate = float(original_data["mocap_frame_rate"])
                else:
                    print(f"Warning: No frame rate found in {motion_path}, skipping")
                    error_count += 1
                    continue
                
                # 计算10秒对应的帧数
                max_segment_frames = int(frame_rate * max_segment_duration)
                
                # 获取总帧数
                if "poses" in original_data:
                    total_frames = len(original_data["poses"])
                elif "num_frames" in original_data:
                    total_frames = int(original_data["num_frames"])
                else:
                    total_frames = len(original_data["trans"])
                
                # 计算总时长
                total_duration = total_frames / frame_rate
                
                # 如果总时长不超过最大段时长，直接复制文件
                if total_duration <= max_segment_duration:
                    # 确保目标文件夹存在
                    os.makedirs(os.path.dirname(new_motion_path), exist_ok=True)
                    
                    # 创建数据副本
                    data = {}
                    for key in original_data.keys():
                        data[key] = original_data[key].copy()
                    
                    # 保存未切分的文件
                    np.savez(new_motion_path, **data)
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count} files...")
                    
                    continue
                
                # 需要切分的文件：计算n等分的分段数量
                # n为可使每一段的长度小于max_segment_duration的最小整数
                num_segments = int(np.ceil(total_duration / max_segment_duration))
                
                # 计算每段的帧数（平均分配）
                segment_frames = total_frames // num_segments
                
                print(f"Segmenting {motion_path}: {total_frames} frames ({total_duration:.1f}s) into {num_segments} equal segments of ~{segment_frames} frames each")
                
                # 提取原始文件名（不含扩展名）
                original_filename = os.path.splitext(os.path.basename(motion_path))[0]
                
                # 切分并保存每个段
                for segment_idx in range(num_segments):
                    # 计算每段的起始和结束帧
                    start_frame = segment_idx * segment_frames
                    if segment_idx == num_segments - 1:  # 最后一段包含剩余所有帧
                        end_frame = total_frames
                    else:
                        end_frame = (segment_idx + 1) * segment_frames
                    
                    # 创建段数据
                    segment_data = {}
                    for key in original_data.keys():
                        if key in ["poses", "trans"]:
                            # 对poses和trans进行切片
                            segment_data[key] = original_data[key][start_frame:end_frame].copy()
                        elif key == "betas":
                            # betas保持不变
                            segment_data[key] = original_data[key].copy()
                        elif key == "gender":
                            # gender保持不变
                            segment_data[key] = original_data[key]
                        elif key in ["mocap_framerate", "mocap_frame_rate"]:
                            # 帧率保持不变
                            segment_data[key] = original_data[key]
                        elif key == "num_frames":
                            # 更新帧数
                            segment_data[key] = end_frame - start_frame
                    
                    # 生成段文件名
                    segment_filename = f"{original_filename}_seg{segment_idx:03d}.npz"
                    segment_path = os.path.join(
                        os.path.dirname(new_motion_path),
                        segment_filename
                    )
                    
                    # 确保目标文件夹存在
                    os.makedirs(os.path.dirname(segment_path), exist_ok=True)
                    
                    # 保存段数据
                    np.savez(segment_path, **segment_data)
                    segmented_count += 1
                
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} files, created {segmented_count} segments...")
                    
            except Exception as e:
                print(f"Error processing {motion_path}: {e}")
                error_count += 1
                continue
        
        print(f"Dataset {dataset_name}: Processed {processed_count} files, Segmented into {segmented_count} segments, Errors: {error_count}, Skipped: {skipped_count}")
    
    print(f"\nSegmentation completed!")
    print(f"Total files processed: {processed_count}")
    print(f"Total segments created: {segmented_count}")
    if error_count > 0:
        print(f"Errors encountered: {error_count}")
    else:
        print("All files processed successfully.")
def dataset_to_y_up_coordinate(base_folder, force_redo=False):
    print("Convert dataset to Y-UP coordinate")
    print(f"Base folder: {base_folder}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 安全验证：确保不会意外修改原始数据
    if not os.path.exists(base_folder):
        raise ValueError(f"Base folder does not exist: {base_folder}")
    
    # 获取根目录下的第一级文件夹（数据集名称）
    dataset_names = []
    for entry in os.listdir(base_folder):
        entry_path = os.path.join(base_folder, entry)
        if os.path.isdir(entry_path) and not entry.startswith('M_'):
            dataset_names.append(entry)
    
    print(f"Found datasets: {dataset_names}")
    
    for dataset_name in dataset_names:
        print(f"\nProcessing dataset: {dataset_name}")
        if dataset_name.lower() != 'lara':
            continue
        
        # 创建存储路径：[{base_folder所在路径}/Processed_{base_folder文件夹名称}/{dataset_name}]
        base_folder_name = os.path.basename(os.path.normpath(base_folder))
        processed_folder_name = f"Processed_{base_folder_name}"
        new_base_folder = os.path.join(os.path.dirname(base_folder), processed_folder_name, dataset_name)
        
        print(f"Processed data will be stored in: {new_base_folder}")
        
        # 使用loop_amams遍历该数据集下的所有npz文件
        dataset_folder = os.path.join(base_folder, dataset_name)
        iterator = loop_amams(
            dataset_folder,
            new_base_folder,
            ext=".npz",
            newext=".npz",
            force_redo=force_redo,
            only_mirror=False,
        )
        
        processed_count = 0
        error_count = 0
        integrity_issues = 0
        skipped_count = 0
        
        for motion_path, new_motion_path in iterator:
            # 安全验证：确保不会处理镜像文件夹中的文件
            
            # 安全验证：确保新文件路径与原始文件路径不同
            if motion_path == new_motion_path:
                print(f"Error: Source and destination paths are the same: {motion_path}")
                error_count += 1
                continue
            
            # 如果force_redo为false且目标文件已存在，则跳过
            if not force_redo and os.path.exists(new_motion_path):
                skipped_count += 1
                if skipped_count % 100 == 0:
                    print(f"Skipped {skipped_count} existing files...")
                continue
            
            try:
                # 加载原始数据（创建深拷贝确保隔离）
                original_data = np.load(motion_path, allow_pickle=True)
                data = {}
                for key in original_data.keys():
                    data[key] = original_data[key].copy()         
                # 处理pose和trans数据，支持poses和pose键名兼容
                poses = torch.from_numpy(data["poses"]).reshape(len(data["poses"]), -1, 3).to(device)
                if poses.shape[1] != 52:
                    if dataset_name.lower() == 'bmlmovi':
                        print('fix BMLmovi pose 55->52')
                        poses = torch.cat([poses[:, :22], poses[:, -30:]], dim=1)
                    else:
                        print(f"Warning: Expected 52 joints in poses, but got {poses.shape[1]}. skip {motion_path}")
                        skipped_count += 1
                        continue
                trans = torch.from_numpy(data["trans"]).to(device)

                R_SMPL = torch.Tensor(
                            [[[1, 0, 0], [0, 0, 1], [0, -1, 0]]]
                        ).to(poses.dtype).to(device)
                # import pdb; pdb.set_trace()
                root_oris = axis_angle_to_rotation_matrix(poses[:, 0])
                root_oris = R_SMPL.matmul(root_oris)
                root_oris = rotation_matrix_to_axis_angle(root_oris).reshape(-1, 3)
                poses[:, 0] = root_oris
                trans = R_SMPL.matmul(trans.unsqueeze(-1)).view_as(trans)
                
                data["poses"] = np.array(poses.flatten(1).cpu())
                data["trans"] = np.array(trans.cpu())
                    
                # 确保目标文件夹存在
                os.makedirs(os.path.dirname(new_motion_path), exist_ok=True)
                
                # 保存转换后数据
                np.savez(new_motion_path, **data)
                
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} files...")
                    
            except Exception as e:
                print(f"Error processing {motion_path}: {e}")
                error_count += 1
                continue
        
        print(f"Dataset {dataset_name}: Processed {processed_count} files, Errors: {error_count}, Integrity issues: {integrity_issues}, Skipped: {skipped_count}")
    
    print("\nY-UP convertion completed!")
    if integrity_issues > 0:
        print(f"WARNING: Found {integrity_issues} potential integrity issues. Check logs for details.")
    else:
        print("All original files appear to be unchanged.")


class SMPLHProcessor:
    def __init__(self, keep_hand=False):
        super().__init__()
        self.pre_transform_R = None
        self.keep_hand = keep_hand
        if keep_hand:
            self.n_j = 52
        else:
            self.n_j = 22
        
        # 初始化数据存储列表
        self.pose = []
        self.joint = []
        self.trans = []
        self.frame_rate = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def clear_cache(self):
        """清空缓存的数据列表"""
        self.pose.clear()
        self.joint.clear()
        self.trans.clear()
        self.frame_rate.clear()


    @torch.no_grad()
    def load_npz(self, path, fps=None, mirror=False):
        try:
            cdata = np.load(path, allow_pickle=True)
        except:
            FileNotFoundError(path)
            return

        motion = smplh_to_body_motion(
            cdata, fps=fps)

        return motion['poses'][:, :self.n_j].cpu(), motion['joint'][:, :self.n_j].cpu(), motion['trans'].cpu(), motion['mocap_framerate'], motion['betas']
    
    def save_to_pt(self, path_load, path_save, fps=30, seq_len=256, max_batch=1000, 
                     include_files=None, exclude_files=None):
        """
        加载并保存SMPL-H运动数据，支持多个包含文件和多个除外文件
        
        参数:
            path_load: 数据加载根目录
            path_save: 数据保存目录
            fps: 目标帧率
            seq_len: 序列长度
            max_batch: 最大批处理大小
            include_files: 包含文件路径列表（txt格式，每行一个相对路径）
            exclude_files: 除外文件路径列表（txt格式，每行一个相对路径）
        """
        print(f"Saving data to {path_save} at {fps} fps...")
        save_dir = path_save
        os.makedirs(save_dir, exist_ok=True)

        # 收集所有要处理的文件路径
        all_file_paths = set()
        exclude_paths = set()
        
        # 处理包含文件
        if include_files:
            for include_file in include_files:
                if os.path.exists(include_file):
                    with open(include_file, 'r') as f:
                        for line in f:
                            file_path = line.strip()
                            if file_path:
                                full_path = os.path.join(path_load, file_path)
                                all_file_paths.add(full_path)
                    print(f"Include {len([line for line in open(include_file) if line.strip()])} paths from {include_file}")
                else:
                    print(f"Warning: Include file not found: {include_file}")
        
        # 如果没有指定包含文件，则处理整个目录
        if not include_files:
            motion_data = []
            for entry in os.listdir(path_load):
                entry_path = os.path.join(path_load, entry)
                if os.path.isdir(entry_path):
                    motion_data.append(entry)
            
            for ds_name in motion_data:
                # 使用递归模式查找所有子目录中的npz文件
                npz_paths = glob(os.path.join(path_load, ds_name, "**", "*.npz"), recursive=True)
                all_file_paths.update(npz_paths)
        
        # 处理除外文件
        if exclude_files:
            for exclude_file in exclude_files:
                if os.path.exists(exclude_file):
                    with open(exclude_file, 'r') as f:
                        for line in f:
                            file_path = line.strip()
                            if file_path:
                                full_path = os.path.join(path_load, file_path)
                                exclude_paths.add(full_path)
                    print(f"Exclude {len([line for line in open(exclude_file) if line.strip()])} paths from {exclude_file}")
                else:
                    print(f"Warning: Exclude file not found: {exclude_file}")
            
            # 从包含文件中排除除外文件
            # import pdb; pdb.set_trace()
            all_file_paths = all_file_paths - exclude_paths
        
        print(f"Total files to process: {len(all_file_paths)} / {len(all_file_paths)+len(exclude_paths)}")
        
        if len(all_file_paths) == 0:
            print("No files found to process")
            return
        
        # 按照数据集名称分组文件
        dataset_files = {}
        for file_path in all_file_paths:
            # 提取数据集名称（路径中的第一级目录）
            rel_path = os.path.relpath(file_path, path_load)
            dataset_name = rel_path.split(os.sep)[0]
            if dataset_name not in dataset_files:
                dataset_files[dataset_name] = []
            dataset_files[dataset_name].append(file_path)
        
        total_idx = 0
        
        for dataset_name, npz_paths in dataset_files.items():
            print(f"\nProcessing dataset: {dataset_name}")
            print(f"Files in dataset: {len(npz_paths)}")
            
            # 按照batch分块处理，避免内存占用过高
            batch_num = (len(npz_paths) + max_batch - 1) // max_batch
            
            for b in range(batch_num):
                start_idx = b * max_batch
                end_idx = min((b + 1) * max_batch, len(npz_paths))
                npz_paths_batch = npz_paths[start_idx:end_idx]
                
                self.clear_cache()
                
                for npz_fname in tqdm(npz_paths_batch, desc=f"Batch {b+1}/{batch_num}"):
                    try:
                        _pose, _joint, _trans, _frame_rate, _beta = self.load_npz(npz_fname, fps=fps)
                        # joint补充trans
                        _joint = _joint + _trans.unsqueeze(1)

                        self.pose.append(_pose)
                        self.joint.append(_joint)
                        self.trans.append(_trans)
                        self.frame_rate.append(_frame_rate)
                        
                    except Exception as e:
                        print(f"Error loading {npz_fname}: {e}")
                        continue
                
                if len(self.pose) > 0:
                    print(f"Loaded {len(self.pose)} sequences from batch {b+1}")
                    
                    # 处理并保存当前批次的序列
                    for i in tqdm(range(len(self.pose)), desc="Saving segments"):
                        _pose = self.pose[i].to(self.device)
                        _joint = self.joint[i].to(self.device)

                        _trans = _joint[:, 0].clone().to(self.device)
                        # 转换为相对root的joint position
                        _joint -= _trans.unsqueeze(1)

                        _pose = axis_angle_to_rotation_matrix(_pose).reshape(-1, self.n_j, 3, 3)

                        num_segments = len(_joint) // seq_len

                        # 保存分段数据
                        for s in range(num_segments):
                            start = s * seq_len
                            end = start + seq_len
                            segment = {
                                'pose': _pose[start:end].cpu(), 
                                'joint': _joint[start:end].cpu(), 
                                'trans': _trans[start:end].cpu()
                            }
                            
                            # 提取子数据集名称（如果有）
                            sub_dataset_name = ""
                            rel_path = os.path.relpath(npz_fname, path_load)
                            path_parts = rel_path.split(os.sep)
                            if len(path_parts) > 2:  # 有子数据集
                                sub_dataset_name = path_parts[1]
                            
                            # 提取原始npz文件名（不含扩展名）
                            original_filename = os.path.splitext(os.path.basename(npz_fname))[0]
                            
                            # 生成新格式的文件名
                            if sub_dataset_name:
                                filename = f"{dataset_name}_{sub_dataset_name}_{original_filename}_{s:06d}.pt"
                            else:
                                filename = f"{dataset_name}_{original_filename}_{s:06d}.pt"
                            
                            torch.save(segment, os.path.join(save_dir, filename))
                            total_idx += 1

                        # 若数据长度小于设置的分段长度，则直接保存
                        if num_segments == 0:
                            segment = {
                                'pose': _pose.cpu(), 
                                'joint': _joint.cpu(), 
                                'trans': _trans.cpu()
                            }
                            filename = f"{dataset_name}_{total_idx:06d}.pt"
                            torch.save(segment, os.path.join(save_dir, filename))
                            total_idx += 1
                    
                    # 清空当前批次数据
                    self.clear_cache()

        print(f"\nTotal {total_idx} joint segments saved to {save_dir}")
        return total_idx

def gather_npz_data(root_path, split_file_path, output_folder, target_fps=30, force_redo=False, skip_mirror=True, gather_meta=False):
    """
    从split文件（如train_amass.txt）读取数据文件路径，转换到指定fps并保存到输出文件夹
    
    参数:
        split_file_path: split文件路径（如train_amass.txt）
        output_folder: 输出文件夹路径
        target_fps: 目标帧率
        force_redo: 是否强制重新处理已存在的文件
        gather_meta: 是否同时拷贝物理误差json文件
    """
    print(f"Gathering data from split file: {split_file_path}")
    print(f"Target FPS: {target_fps}")
    print(f"Output folder: {output_folder}")
    print(f"Gather meta files: {gather_meta}")
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 读取split文件
    if not os.path.exists(split_file_path):
        raise ValueError(f"Split file does not exist: {split_file_path}")
    
    with open(split_file_path, 'r') as f:
        file_paths = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(file_paths)} files in split file")
    
    processed_count = 0
    error_count = 0
    skipped_count = 0
    meta_copied_count = 0
    
    for file_path in tqdm(file_paths, desc="Processing files"):
        # 构建完整文件路径
        full_path = os.path.join(root_path, file_path)
        if file_path.find('M_') >= 0 and skip_mirror:
            # print(f'skip mirrored data {full_path}')
            continue
        
        # 检查文件是否存在
        if not os.path.exists(full_path):
            print(f"Warning: File not found: {full_path}")
            error_count += 1
            continue
        
        # 生成输出文件名（保持原始文件名结构）
        output_filename = os.path.basename(file_path)
        output_path = os.path.join(output_folder, output_filename)
        
        # 检查是否已存在且不需要重新处理
        if not force_redo and os.path.exists(output_path):
            skipped_count += 1
            continue
        
        # try:
        # 使用smplh_to_body_motion加载动作数据
        smpl_dict = np.load(full_path, allow_pickle=True)
        if smpl_dict["poses"].shape[1] != 52 * 3:
            print(f"Warning: Expected pose shape (N, 156), but got {smpl_dict['poses'].shape} in file {full_path}. Skipping.")
            error_count += 1
            continue
        motion_data = smplh_to_body_motion(smpl_dict, fps=target_fps)

        # import pdb; pdb.set_trace()
        
        # 使用motion2npz_dict转换为npz格式
        npz_data = motion2npz_dict(
            pose=motion_data["poses"],
            trans=motion_data["trans"],
            frame_rate=target_fps,
            betas=motion_data["betas"],
            gender=motion_data["gender"]
        )
        
        if npz_data is None:
            print(f"Error: Failed to convert motion data for {file_path}")
            error_count += 1
            continue
        
        # 确保输出目录存在
        output_dir = os.path.dirname(os.path.join(output_folder, file_path))
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存转换后的数据
        np.savez(os.path.join(output_folder, file_path), **npz_data)
        
        processed_count += 1
        
        # 如果启用了gather_meta，查找并拷贝物理误差json文件
        if gather_meta:
            # 根据calc_phys函数的实现，meta_data文件存储在meta_data文件夹中
            npz_dir = os.path.dirname(full_path)
            meta_data_dir = os.path.join(npz_dir, "meta_data")
            json_src_path = os.path.join(meta_data_dir, "phys_metrics.json")
            
            if os.path.exists(json_src_path):
                # 拷贝json文件到输出目录
                output_meta_dir = os.path.join(output_dir, "meta_data")
                os.makedirs(output_meta_dir, exist_ok=True)
                json_dst_path = os.path.join(output_meta_dir, "phys_metrics.json")
                
                # 检查目标文件是否已存在，避免重复拷贝
                if not os.path.exists(json_dst_path):
                    try:
                        import shutil
                        shutil.copy2(json_src_path, json_dst_path)
                        meta_copied_count += 1
                        if meta_copied_count % 100 == 0:
                            print(f"Copied {meta_copied_count} meta files...")
                    except Exception as e:
                        print(f"Warning: Failed to copy meta file {json_src_path}: {e}")
                else:
                    # 如果文件已存在，增加计数但跳过拷贝
                    meta_copied_count += 1
            else:
                print(f"Info: No meta file found at {json_src_path}")
        
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} files...")
                
        # except Exception as e:
        #     print(f"Error processing {file_path}: {e}")
        #     error_count += 1
        #     continue
    
    print(f"\nGathering completed!")
    print(f"Processed: {processed_count}")
    print(f"Skipped (already exists): {skipped_count}")
    print(f"Errors: {error_count}")
    if gather_meta:
        print(f"Meta files copied: {meta_copied_count}")
    
    return {
        "processed_count": processed_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
        "meta_copied_count": meta_copied_count if gather_meta else 0
    }
