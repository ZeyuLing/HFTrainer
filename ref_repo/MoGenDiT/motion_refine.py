#!/usr/bin/env python
"""
命令行版的Motion Refine脚本

用法示例:
python motion_refine.py --ckpt-dir ./save/ckpt --model-name MoreDiff-0.1B --step 60000 --denoise-step 10 \
    --input-dir ./data/collected_test_data \
    --output-root ./data \
    --mode denoise
"""

import torch
import numpy as np
import argparse
import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from EasyDiffusion import GaussianDiffusion, BetaSchedule, ModelMeanType
from trainer.data_loader import NpzMotion
from motion_process.motion_representation import OccamMotionRep
from motion_process.smplh_processor import motion2npz_dict
from model.more_diff import get_MoreDiff_model
from motion_process.motion_refiner import MoreDiffRefiner


class ModelConfigLoader:
    """模型配置加载器，用于从train_args.json读取配置"""
    
    @staticmethod
    def load_train_args(model_dir: Path) -> Dict:
        """
        从模型目录加载train_args.json配置文件
        
        Args:
            model_dir: 模型目录路径（包含train_args.json）
            
        Returns:
            Dict: 训练配置字典
        """
        train_args_path = model_dir / "train_args.json"
        
        if not train_args_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {train_args_path}")
        
        with open(train_args_path, 'r') as f:
            config = json.load(f)
        
        return config
    
    @staticmethod
    def find_model_files(model_dir: Path) -> Tuple[List[str], List[str]]:
        """
        查找模型目录中的所有模型权重文件
        
        Args:
            model_dir: 模型目录路径
            
        Returns:
            Tuple[List[str], List[str]]: (常规模型文件列表, EMA模型文件列表)
        """
        model_files = []
        ema_model_files = []
        
        # 查找所有model_*.pth文件
        model_pattern = str(model_dir / "model_*.pth")
        model_files = sorted(glob.glob(model_pattern))
        
        # 查找所有ema_model_*.pth文件
        ema_pattern = str(model_dir / "ema_model_*.pth")
        ema_model_files = sorted(glob.glob(ema_pattern))
        
        return model_files, ema_model_files
    
    @staticmethod
    def get_step_from_filename(filename: str) -> int:
        """
        从文件名中提取训练步数
        
        Args:
            filename: 模型文件名（如model_0000010000.pth）
            
        Returns:
            int: 训练步数
        """
        # 提取数字部分
        base_name = Path(filename).stem
        # 移除前缀
        if base_name.startswith("model_"):
            step_str = base_name[6:]  # 移除"model_"
        elif base_name.startswith("ema_model_"):
            step_str = base_name[10:]  # 移除"ema_model_"
        else:
            raise ValueError(f"无法识别的模型文件名格式: {filename}")
        
        # 转换为整数
        try:
            step = int(step_str)
        except ValueError:
            raise ValueError(f"文件名中的步数格式不正确: {filename}")
        
        return step
    
    @staticmethod
    def find_model_by_step(model_dir: Path, step: Optional[int] = None, use_ema: bool = False) -> str:
        """
        根据步数查找模型文件
        
        Args:
            model_dir: 模型目录路径
            step: 指定的步数，如果为None则使用最新的模型
            use_ema: 是否使用EMA模型
            
        Returns:
            str: 模型文件路径
        """
        model_files, ema_model_files = ModelConfigLoader.find_model_files(model_dir)
        
        # 选择文件列表
        if use_ema:
            file_list = ema_model_files
            prefix = "ema_model_"
        else:
            file_list = model_files
            prefix = "model_"
        
        if not file_list:
            raise FileNotFoundError(f"在目录 {model_dir} 中没有找到{prefix}*.pth文件")
        
        if step is None:
            # 使用最新的模型（文件名数字最大的）
            latest_file = max(file_list, key=lambda x: ModelConfigLoader.get_step_from_filename(x))
            print(f"使用最新的模型: {Path(latest_file).name}")
            return latest_file
        else:
            # 查找指定步数的模型
            target_file = None
            for file_path in file_list:
                file_step = ModelConfigLoader.get_step_from_filename(file_path)
                if file_step == step:
                    target_file = file_path
                    break
            
            if target_file is None:
                available_steps = [ModelConfigLoader.get_step_from_filename(f) for f in file_list]
                raise FileNotFoundError(
                    f"未找到步数为 {step} 的模型文件。可用步数: {available_steps}"
                )
            
            print(f"使用指定步数的模型: {Path(target_file).name}")
            return target_file


def traverse_subfolders(root_dir: str) -> List[str]:
    """
    遍历 root_dir 下的所有包含.npz文件的文件夹
    
    Args:
        root_dir: 根目录路径
        
        Returns:
        List[str]: 包含npz文件的文件夹路径列表
    """
    root_path = Path(root_dir)
    
    # 检查根目录是否存在
    if not root_path.is_dir():
        print(f"错误：目录不存在 - {root_dir}")
        return []

    # 存储所有包含npz文件的文件夹路径
    npz_folders = []

    # 递归查找所有.npz文件
    for npz_path in root_path.rglob("*.npz"):
        folder_path = npz_path.parent
        # 将文件夹路径转换为相对于root_dir的路径
        relative_folder = folder_path.relative_to(root_path)
        folder_str = str(root_path / relative_folder)
        
        # 避免重复添加
        if folder_str not in npz_folders:
            npz_folders.append(folder_str)
    
    # 如果没有找到任何npz文件，返回根目录
    if len(npz_folders) == 0:
        npz_folders.append(str(root_dir))
        print(f"警告：在 {root_dir} 中未找到.npz文件，将使用根目录")

    return npz_folders

def refine_motion(
    motion: torch.Tensor,
    refiner,
    motion_rep,
    device,
    denoise_step: int,
    mode: str,
    fast_sampling: bool = True,
    imputation_mode: str = "skip_last",
) -> torch.Tensor:
    """
    Core motion refinement: normalization -> refinement -> decode.
    
    Args:
        motion: Encoded motion tensor (seq_len, data_dim)
        refiner: MoreDiffRefiner instance
        motion_rep: Motion representation encoder
        device: Compute device
        denoise_step: Number of denoising steps
        mode: Refinement mode ('denoise', 'ada_denoise', 'trans_regen')
        fast_sampling: Whether to use fast sampling
        imputation_mode: Imputation mode for diffusion sampling.
            'skip_last': apply imputation at every step except the last (default)
            'all': apply imputation at every step including the last
            'none': disable imputation entirely
        
    Returns:
        Tuple of (pose, joint, trans) decoded from refined motion
    """
    # Normalization
    motion = motion_rep.normalization(motion.to(device))
    origin_motion = motion.unsqueeze(0).to(device)
    
    # Refinement
    gen_motion = refiner.refine(
        motion=origin_motion,
        cond=None,
        step=denoise_step,
        mode=mode,
        use_windowed=True,
        fast_sampling=fast_sampling,
        imputation_mode=imputation_mode,
    )

    # Decode
    pose, joint, trans = motion_rep.decode(gen_motion[0])
    
    return pose, joint, trans


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Motion Refine 命令行工具')
    
    # 必需参数
    parser.add_argument('--ckpt-dir', type=str, required=True,
                       help='模型检查点根目录，如: ./save/ckpt')
    parser.add_argument('--model-name', type=str, required=True,
                       help='模型名称，如: MoreDiff-0.1B')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='输入数据根目录，包含npz文件')
    parser.add_argument('--output-root', type=str, required=True,
                       help='输出根目录，实际的输出目录会自动构建为: output-root/[input-dir-name]_[model-name]_[mode]_[denoise-step]')
    
    # 模型选择参数
    parser.add_argument('--step', type=int, default=None,
                       help='模型训练步数，如果未指定则使用最新模型')
    parser.add_argument('--use-ema', action='store_true',
                       help='是否使用EMA模型')
    
    # 修复参数
    parser.add_argument('--denoise-step', type=int, default=10,
                       help='去噪步数，默认: 10')
    parser.add_argument('--mode', type=str, default='denoise',
                       choices=['denoise', 'ada_denoise', 'trans_regen'],
                       help='修复模式，默认: denoise')
    parser.add_argument('--fast-sampling', action='store_true', default=True,
                       help='是否使用快速采样（仅对trans_regen模式生效），默认: True')
    parser.add_argument('--no-fast-sampling', dest='fast_sampling', action='store_false',
                       help='禁用快速采样，使用完整的1000步DDIM采样')
    parser.add_argument('--imputation-mode', type=str, default='skip_last',
                       choices=['skip_last', 'all', 'none'],
                       help='插补模式: skip_last(跳过最后一步插补,默认), all(每步都插补), none(不插补)')
    
    # 其他参数（从train_args.json读取，但允许覆盖）
    parser.add_argument('--global-pose', type=lambda x: (str(x).lower() == 'true'), default=None,
                       help='是否使用全局姿态，如果不指定则从train_args.json读取')
    parser.add_argument('--keep-hand', action='store_true',
                       help='是否保留手部姿态')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='计算设备，默认: cuda:0')
    parser.add_argument('--skip-existing', action='store_true',
                       help='跳过已存在的输出文件')
    
    args = parser.parse_args()
    
    # 验证参数
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"检查点目录不存在: {ckpt_dir}")
    
    model_dir = ckpt_dir / args.model_name
    if not model_dir.exists():
        # 列出可用的模型
        available_models = [d.name for d in ckpt_dir.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"模型目录不存在: {model_dir}\n"
            f"可用的模型: {available_models}"
        )
    
    if not Path(args.input_dir).exists():
        raise FileNotFoundError(f"输入目录不存在: {args.input_dir}")
    
    # 构建输出目录路径
    # output-root/[input-dir名称]_{model-name}_{mode}_{denoise-step}
    input_dir_name = Path(args.input_dir).name
    output_dir_name = f"{input_dir_name}_{args.model_name}_{args.mode}_{args.denoise_step}"
    output_dir = Path(args.output_root) / output_dir_name
    
    # 创建输出根目录和实际输出目录
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"输出根目录: {args.output_root}")
    print(f"实际输出目录: {output_dir}")
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型配置
    print(f"加载模型配置: {args.model_name}")
    config_loader = ModelConfigLoader()
    
    # 读取train_args.json
    try:
        train_args = config_loader.load_train_args(model_dir)
        print(f"从配置文件读取的配置:")
        print(f"  model_version: {train_args.get('model_version', '未指定')}")
        print(f"  global_pose: {train_args.get('global_pose', '未指定')}")
    except Exception as e:
        print(f"警告: 无法读取train_args.json: {e}")
        print(f"将使用命令行参数或默认值")
        train_args = {}
    
    # 确定最终配置参数
    # model_version: 优先使用train_args.json中的，如果没有则使用默认值2
    model_version = train_args.get('model_version', 2)
    
    # global_pose: 优先使用命令行参数，其次使用train_args.json，最后使用False
    if args.global_pose is not None:
        global_pose = args.global_pose
        print(f"使用命令行指定的global_pose: {global_pose}")
    elif 'global_pose' in train_args:
        global_pose = train_args['global_pose']
        print(f"使用配置文件中的global_pose: {global_pose}")
    else:
        global_pose = False
        print(f"使用默认global_pose: {global_pose}")
    
    # 查找模型文件
    print(f"查找模型权重文件...")
    try:
        model_file = config_loader.find_model_by_step(
            model_dir=model_dir,
            step=args.step,
            use_ema=args.use_ema
        )
        print(f"找到模型文件: {model_file}")
    except Exception as e:
        raise FileNotFoundError(f"找不到模型权重文件: {e}")
    
    # 初始化运动表示和处理器
    motion_rep = OccamMotionRep(
        keep_hand=args.keep_hand, 
        global_pose=global_pose
    )
    
    # 加载模型
    print(f"加载模型: {args.model_name}, 版本: {model_version}")
    model = get_MoreDiff_model(
        data_dim=motion_rep.data_dim, 
        version=model_version
    ).to(device)
    
    print(f"加载检查点: {model_file}")
    model.restore(checkpoint_path=model_file)
    model.eval()
    
    # 初始化扩散模型
    diffusion = GaussianDiffusion(
        num_timesteps=1000,
        beta_schedule=BetaSchedule.COSINE,
        model_mean_type=ModelMeanType.START_X,
    )
    
    # 查找所有包含npz文件的文件夹
    folders = traverse_subfolders(args.input_dir)
    print(f"找到 {len(folders)} 个包含npz文件的文件夹")

    # 创建refiner
    refiner = MoreDiffRefiner(motion_rep, model, diffusion)
    
    # 统计信息
    total_files = 0
    processed_files = 0
    skipped_files = 0
    err_files = 0
    
    # 遍历所有文件夹
    for folder_idx, folder in enumerate(folders, 1):
        folder_name = Path(folder).name
        print(f"\n处理文件夹 [{folder_idx}/{len(folders)}]: {folder_name}")
        
        # 一次性加载整个文件夹的数据
        print(f"  加载文件夹数据...")
        try:
            data_dict = NpzMotion.load_data(
                folder,
            )
            
            if not data_dict or len(data_dict.get("file_name", [])) == 0:
                print(f"  文件夹中没有有效数据，跳过")
                continue
                
            total_files_in_folder = len(data_dict["file_name"])
            print(f"  成功加载 {total_files_in_folder} 个文件的数据")
            
        except Exception as e:
            print(f"  加载文件夹数据失败: {str(e)}")
            # 保存错误信息
            error_log_path = output_dir / "error_log.txt"
            with open(error_log_path, "a") as f:
                f.write(f"{folder}: 数据加载失败 - {str(e)}\n")
            continue
        
        # Create NpzMotion object once per folder (avoid repeated instantiation)
        npz_motion = NpzMotion(
            data=data_dict,
            motion_rep=motion_rep,
        )
        
        # 查找文件夹中的所有npz文件
        folder_path = Path(folder)
        npz_files = list(folder_path.rglob("*.npz"))
        
        # 验证加载的文件数量与实际文件数量一致
        loaded_file_count = len(data_dict["file_name"])
        actual_file_count = len(npz_files)
        
        if loaded_file_count != actual_file_count:
            print(f"  警告: 加载的文件数({loaded_file_count})与实际文件数({actual_file_count})不一致")
        
        # 统计信息
        processed_in_folder = 0
        skipped_in_folder = 0
        
        # 遍历文件夹中的所有文件
        for file_idx, npz_file in enumerate(npz_files, 1):
            total_files += 1
            
            # 计算输出路径（保持目录结构）
            relative_path = npz_file.relative_to(args.input_dir)
            output_file_path = output_dir / relative_path
            
            # 检查是否跳过已存在的文件
            if args.skip_existing and output_file_path.exists():
                print(f"  [{file_idx}/{len(npz_files)}] 跳过已存在文件: {npz_file.name}")
                skipped_files += 1
                skipped_in_folder += 1
                continue
            
            # 确保输出目录存在
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # try:
            print(f"  [{file_idx}/{len(npz_files)}] 处理: {npz_file.name}")
            
            # Find the index for current file in data_dict
            file_name = npz_file.name
            if file_name not in data_dict["file_name"]:
                print(f"    警告: 文件 {file_name} 不在加载的数据中，跳过")
                skipped_files += 1
                skipped_in_folder += 1
                continue
            
            idx = data_dict["file_name"].index(file_name)
            
            # Get encoded motion from pre-created NpzMotion object
            motion, length = npz_motion[idx]
            
            # Core refinement: normalization -> refine -> decode
            pose, joint, trans = refine_motion(
                motion=motion,
                refiner=refiner,
                motion_rep=motion_rep,
                device=device,
                denoise_step=args.denoise_step,
                mode=args.mode,
                fast_sampling=args.fast_sampling,
                imputation_mode=args.imputation_mode,
            )
            
            # Concatenate hand pose back
            hand_pose = data_dict["hand_pose"][idx]
            pose = torch.cat([pose, hand_pose.to(pose.device)], dim=1)
            
            # Convert to npz format
            out_npz = motion2npz_dict(
                pose=pose,
                trans=trans,
                frame_rate=30,
                betas=data_dict["beta"][idx],
                gender=data_dict["gender"][idx],
            )
            
            # 保存结果
            np.savez(str(output_file_path), **out_npz)
            print(f"    保存到: {output_file_path}")
            processed_files += 1
            processed_in_folder += 1
                
            # except Exception as e:
            #     print(f"    处理文件 {npz_file.name} 时出错: {str(e)}")
            #     err_files += 1
            #     # 保存错误信息
            #     error_log_path = output_dir / "error_log.txt"
            #     with open(error_log_path, "a") as f:
            #         f.write(f"{npz_file}: {str(e)}\n")
        
        # 打印文件夹处理统计
        print(f"  文件夹处理完成: 已处理 {processed_in_folder}, 已跳过 {skipped_in_folder}")
    
    # 打印统计信息
    print(f"\n{'='*50}")
    print(f"处理完成!")
    print(f"总文件数: {total_files}")
    print(f"已处理: {processed_files}")
    print(f"已跳过: {skipped_files}")
    print(f"输出根目录: {args.output_root}")
    print(f"实际输出目录: {output_dir}")
    
    if err_files > 0:
        print(f"警告: {err_files}个文件处理失败，详情见 {output_dir}/error_log.txt")


if __name__ == "__main__":
    main()