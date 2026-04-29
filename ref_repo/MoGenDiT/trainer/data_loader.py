import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
from articulate.math.angular import axis_angle_to_rotation_matrix

class AdHocMotionData(Dataset):
    @staticmethod
    def load_data(data_root, min_len=30):
        data_root = Path(data_root)
        cache_file = data_root / f"data_cache_minlen{min_len}.npz"
        
        if cache_file.exists():
            print(f"[INFO] Loading cached data from {cache_file}")
            try:
                cache_data = np.load(cache_file, allow_pickle=True)
                data_dict = cache_data['data_dict'].item()
                
                # 兼容旧版缓存格式：如果存在data_root字段，需要将相对路径转换为绝对路径
                if "data_root" in data_dict:
                    print(f"[INFO] Converting old cache format to new format")
                    data_root = Path(data_dict["data_root"])
                    file_paths = data_dict.get("file_paths", [])
                    # 将相对路径转换为绝对路径
                    absolute_paths = [str(data_root / path) for path in file_paths]
                    data_dict["file_paths"] = absolute_paths
                    # 移除data_root字段
                    if "data_root" in data_dict:
                        del data_dict["data_root"]
                
                print(f"[INFO] Loaded {len(data_dict.get('file_paths', []))} files from cache")
                return data_dict
            except Exception as e:
                print(f"[WARNING] Failed to load cache: {e}, regenerating...")
        
        pt_files = list(data_root.rglob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {data_root}")
        
        print(f"[INFO] Found {len(pt_files)} .pt files in {data_root}")
        
        data_dict = {
            "file_paths": [],
        }
        
        valid_files = 0
        for pt_file in tqdm(pt_files):
            try:
                data = torch.load(pt_file, map_location='cpu')
                
                if not isinstance(data, dict):
                    print(f"[WARNING] File {pt_file.name} does not contain a dict, skipping")
                    continue
                    
                required_fields = ['pose', 'joint', 'trans']
                if not all(field in data for field in required_fields):
                    print(f"[WARNING] File {pt_file.name} missing required fields, skipping")
                    continue
                    
                pose_len = len(data['pose'])
                if pose_len < min_len:
                    print(f"[WARNING] File {pt_file.name} too short ({pose_len} < {min_len}), skipping")
                    continue
                
                # 存储完整绝对路径
                data_dict["file_paths"].append(str(pt_file))
                
            except Exception as e:
                print(f"[WARNING] Failed to load {pt_file.name}: {e}, skipping")
                continue
        
        data_dict["valid_files"] = valid_files
        print(f"[INFO] Valid files: {valid_files}/{len(pt_files)}")
        
        try:
            np.savez(cache_file, data_dict=data_dict)
            print(f"[INFO] Data cache saved to {cache_file}")
        except Exception as e:
            print(f"[WARNING] Failed to save cache: {e}")
        
        return data_dict

    @staticmethod
    def merge(data_dict1, data_dict2):
        return {
            "file_paths": data_dict1.get("file_paths", []) + data_dict2.get("file_paths", []),
        }

    def __init__(self, data, motion_rep, fix_len=224):
        super().__init__()
        
        self.file_paths = data["file_paths"]
        self.total_files = len(self.file_paths)
        
        self.motion_rep = motion_rep
        self.fix_len = fix_len
        
        print(f"[INFO] AdHocMotionData initialized with {self.total_files} files, fix_len={fix_len}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """
        获取单个数据样本
        
        返回:
            motion: 处理后的运动数据 (fix_len, data_dim)
            length: 实际有效长度
            cond: 条件信息，这里返回None
        """
        # 获取文件路径（现在是完整绝对路径）
        full_path = Path(self.file_paths[idx])
        
        # 加载原始数据
        try:
            raw_data = torch.load(full_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Failed to load {full_path}: {e}")
        
        # 提取必需字段
        pose_data = raw_data['pose']  # 形状: (seq_len, pose_dim)
        joint_data = raw_data['joint']  # 形状: (seq_len, joint_dim)
        trans_data = raw_data['trans']  # 形状: (seq_len, trans_dim)
        
        seq_len = len(pose_data)
        
        # 随机选择起始帧
        max_start = max(0, seq_len - 60)
        start_frame = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        
        # 计算实际提取的长度（不超过fix_len）
        actual_len = min(self.fix_len, seq_len - start_frame)
        
        # 提取数据片段
        pose_slice = pose_data[start_frame:start_frame + actual_len]
        joint_slice = joint_data[start_frame:start_frame + actual_len]
        trans_slice = trans_data[start_frame:start_frame + actual_len]
        
        # 使用motion_rep进行编码（在CPU上完成所有预处理，避免CUDA重新初始化错误）
        # 注意：在DataLoader的worker进程中使用.to(self.device)会导致CUDA重新初始化错误
        # 因此所有预处理都在CPU上完成，返回CPU数据，由主进程负责移动到CUDA
        motion = self.motion_rep.encode(pose=pose_slice, joint=joint_slice, trans=trans_slice)
        
        # 归一化处理（在CPU上完成）
        motion = self.motion_rep.normalization(motion, height_reset=True)
        
        
        # 填充到固定长度
        if motion.shape[0] < self.fix_len:
            padding_len = self.fix_len - motion.shape[0]
            padding = torch.zeros_like(motion[:1]).repeat(padding_len, 1)  # 形状: (padding_len, data_dim)
            motion = torch.cat([motion, padding], dim=0)
        
        # 返回格式: motion, length, cond (cond为None)
        return motion, actual_len


class NpzMotion(Dataset):
    """
    自动加载npz格式的SMPL-H运动数据集
    与motion_refine.py中的接口保持一致性
    """
    
    @staticmethod
    def load_data(data_root, fps=30):
        """
        加载数据目录下的所有npz文件
        
        参数:
            data_root: 数据根目录
            min_len: 最小序列长度
            motion_rep: 运动表示编码器（用于获取身体关节数）
            
        返回:
            data_dict: 包含file_name, beta, hand_pose, gender等字段的字典
        """
        data_root = Path(data_root)
        
        # 扫描所有npz文件
        npz_files = list(data_root.rglob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {data_root}")
        
        print(f"[INFO] Found {len(npz_files)} .npz files in {data_root}")
        
        data_dict = {
            "file_name": [],
            "poses": [],
            "joint": [],
            "trans": [],
            "beta": [],
            "hand_pose": [],
            "gender": [],
        }
        from motion_process.smplh_processor import smplh_to_body_motion
        
        valid_files = 0
        for npz_file in tqdm(npz_files):
            try:
                # 检查文件是否包含'poses'键
                # import pdb; pdb.set_trace()
                with np.load(npz_file, allow_pickle=True) as data:
                    # 处理可能的smpl_data包装
                    if 'smpl_data' in data:
                        data = data['smpl_data']
                    
                    if 'poses' not in data:
                        print(f"[WARNING] File {npz_file.name} does not contain 'poses' key, skipping")
                        continue
                    
                    motion = smplh_to_body_motion(smplh_data=data, fps=fps)

                    pose = motion["poses"]
                    pose = axis_angle_to_rotation_matrix(pose).reshape(pose.shape[0], -1, 3, 3)

                    data_dict["file_name"].append(npz_file.name)
                    data_dict["poses"].append(pose[:, :22])
                    data_dict["joint"].append(motion["joint"][:, :22])
                    data_dict["trans"].append(motion["trans"])
                    data_dict["beta"].append(motion["betas"])
                    data_dict["hand_pose"].append(pose[:, 22:])
                    data_dict["gender"].append(motion["gender"])
                    
            except Exception as e:
                print(f"[WARNING] Failed to load {npz_file.name}: {e}, skipping")
                continue
        
        print(f"[INFO] Valid npz files: {valid_files}/{len(npz_files)}")
        
        return data_dict
    
    def __init__(self, data, motion_rep):
        """
        初始化NpzMotion数据集
        
        参数:
            data: 数据字典，包含file_name, beta, hand_pose, gender等字段
            motion_rep: 运动表示编码器
            fix_len: 固定序列长度（默认600）
            use_vel: 是否使用速度信息（根据motion_refine.py中的调用）
        """
        super().__init__()
        
        self.file_names = data["file_name"]
        self.pose = data["poses"]
        self.joint = data["joint"]
        self.trans = data["trans"]
        self.betas = data["beta"]
        self.hand_poses = data["hand_pose"]
        self.genders = data["gender"]
        
        self.motion_rep = motion_rep
        
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        """
        获取单个数据样本
        
        返回:
            motion: 处理后的运动数据 (fix_len, data_dim)
            length: 实际有效长度
            cond: 条件信息，这里返回None以保持与motion_refine.py的兼容性
        """
        # 获取文件名和相关数据
        pose, joint, trans = self.pose[index], self.joint[index], self.trans[index]
       
        motion = self.motion_rep.encode(pose=pose, joint=joint, trans=trans)
        
        return motion, motion.shape[0]