# 示例调用：python -m sample.degradation_batch --config ./sample/degradation_settings.json

import torch
import numpy as np
import os
from pathlib import Path
from motion_process.motion_degradation import GlobalMotionDegradation
from my_data.amass import NpzMotion
from motion_process.motion_representation import OccamMotionRep
from motion_process.processors import Npz2VecProcessor
import argparse
import json


def traverse_subfolders(root_dir):
    """遍历 root_dir 下的所有末端文件夹（无任何子文件夹的文件夹）"""
    root_path = Path(root_dir)
    # 检查根目录是否存在
    if not root_path.is_dir():
        print(f"错误：目录不存在 - {root_dir}")
        return []

    # 存储所有末端文件夹路径
    end_subfolders = []

    # 递归匹配所有路径，过滤出文件夹（排除根目录本身）
    for path in root_path.rglob("*"):
        if path.is_dir() and path != root_path:
            # 检查当前文件夹是否有子文件夹
            # 遍历当前文件夹下的所有条目，判断是否存在子文件夹
            has_subfolders = any(p.is_dir() for p in path.iterdir())

            # 如果没有子文件夹，则是末端文件夹，加入列表
            if not has_subfolders:
                end_subfolders.append(str(path))
    if len(end_subfolders) == 0:
        end_subfolders.append(str(root_dir))

    return end_subfolders


def load_args(json_path):
    """从JSON文件加载参数设置"""
    try:
        with open(json_path, "r") as f:
            config_dict = json.load(f)

        # 处理嵌套的配置文件结构
        args_dict = {}

        # 尝试从degradation_settings键获取配置
        if "degradation_settings" in config_dict:
            settings = config_dict["degradation_settings"]

            # 扁平化嵌套结构
            if "input_root" in settings:
                args_dict["input_root"] = settings["input_root"]
            if "output_root" in settings:
                args_dict["output_root"] = settings["output_root"]
            if "dataset_name" in settings:
                args_dict["dataset_name"] = settings["dataset_name"]

            # 处理degradation_parameters
            if "degradation_parameters" in settings:
                params = settings["degradation_parameters"]
                for key in [
                    "min_segments",
                    "max_segments",
                    "min_segment_length",
                    "max_segment_length",
                    "strength_multiplier",
                    "seed",
                ]:
                    if key in params:
                        args_dict[key] = params[key]

            # 处理system设置
            if "system" in settings:
                sys_config = settings["system"]
                if "use_vel" in sys_config:
                    args_dict["use_vel"] = sys_config["use_vel"]
                if "keep_hand" in sys_config:
                    args_dict["keep_hand"] = sys_config["keep_hand"]
                if "global_pose" in sys_config:
                    args_dict["global_pose"] = sys_config["global_pose"]
        else:
            # 如果没有degradation_settings键，假设扁平结构
            args_dict = config_dict

        # 创建命名空间对象
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        return Args(**args_dict)

    except json.JSONDecodeError as e:
        print(f"JSON解析错误 {json_path}: {e}")
        return None
    except Exception as e:
        print(f"加载配置文件错误 {json_path}: {e}")
        return None


def main():
    """主函数：批量应用随机退化到动作文件"""
    # 配置参数
    parser = argparse.ArgumentParser(description="批量动作退化处理")
    parser.add_argument(
        "--config",
        type=str,
        default="./sample/degradation_settings.json",
        help="配置JSON文件路径",
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default=None,
        help="输入动作文件根目录，优先级高于配置文件",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./data/degraded_motions",
        help="输出文件根目录",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="degraded_dataset",
        help="数据集名称，用于组织输出目录",
    )
    parser.add_argument(
        "--min_segments", type=int, default=2, help="随机退化段落的最小数量"
    )
    parser.add_argument(
        "--max_segments", type=int, default=5, help="随机退化段落的最大数量"
    )
    parser.add_argument(
        "--min_segment_length", type=int, default=10, help="段落最小长度（帧数）"
    )
    parser.add_argument(
        "--max_segment_length", type=int, default=30, help="段落最大长度（帧数）"
    )
    parser.add_argument(
        "--strength_multiplier", type=float, default=1.0, help="退化强度乘子"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出文件")

    parser.add_argument(
        "--use_vel",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="是否使用速度信息（true/false）",
    )
    parser.add_argument(
        "--keep_hand",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="是否保留手部关节（true/false）",
    )
    parser.add_argument(
        "--global_pose",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="是否使用全局姿态（true/false）",
    )

    cmd_args = parser.parse_args()

    # 从配置文件加载默认参数（如果存在）
    if os.path.exists(cmd_args.config):
        print(f"加载配置文件: {cmd_args.config}")
        config_args = load_args(cmd_args.config)

        if config_args is not None:
            # 使用配置文件的参数作为默认值（只有当命令行参数未设置时才使用）
            config_keys = [
                "input_root",
                "output_root",
                "dataset_name",
                "min_segments",
                "max_segments",
                "seed",
                "use_vel",
                "keep_hand",
                "global_pose",
            ]

            for key in config_keys:
                if hasattr(config_args, key):
                    config_value = getattr(config_args, key)
                    cmd_value = getattr(cmd_args, key, None)

                    # 如果命令行参数未设置，或者命令行参数是默认值，则使用配置值
                    if key == "input_root" and cmd_value is None:
                        setattr(cmd_args, key, config_value)
                    elif key == "overwrite" or key == "config":
                        # 这些参数不进行覆盖
                        pass
                    else:
                        # 检查是否是默认值
                        is_default_value = False
                        for action in parser._actions:
                            if action.dest == key:
                                if cmd_value == action.default:
                                    is_default_value = True
                                break

                        if is_default_value:
                            setattr(cmd_args, key, config_value)

            print(f"从配置文件加载了 {len(config_keys)} 个参数")
        else:
            print(f"警告：无法加载配置文件 {cmd_args.config}")
    else:
        print(f"配置文件不存在，使用默认参数: {cmd_args.config}")

    # 验证必要参数
    if cmd_args.input_root is None:
        print("错误：必须指定输入目录（通过--input_root参数或配置文件）")
        return
    # 验证必要参数
    if cmd_args.input_root is None:
        print("错误：必须指定输入目录（通过--input_root参数或配置文件）")
        return

    # 设置随机种子以确保可重现性
    torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化退化处理器
    degradation = GlobalMotionDegradation(device=device)
    print("退化处理器初始化完成")

    # 初始化动作表示和处理工具
    motion_rep = OccamMotionRep(
        keep_hand=cmd_args.keep_hand, global_pose=cmd_args.global_pose
    )
    mp = Npz2VecProcessor(keep_hand=cmd_args.keep_hand)

    # 检查输入目录是否存在
    if not Path(cmd_args.input_root).exists():
        print(f"错误：输入目录不存在 - {cmd_args.input_root}")
        return

    # 递归查找所有文件夹
    folders = traverse_subfolders(cmd_args.input_root)
    print(f"找到 {len(folders)} 个文件夹需要处理")

    # 创建输出目录结构
    output_path = os.path.join(cmd_args.output_root, cmd_args.dataset_name)
    os.makedirs(output_path, exist_ok=True)
    print(f"输出目录: {output_path}")

    # 记录处理统计信息
    total_files = 0
    processed_files = 0
    skipped_files = 0

    # 处理每个文件夹
    for folder_idx, sub_folder in enumerate(folders):
        sub_folder_name = sub_folder.split("/")[-1]
        print(f"[{folder_idx+1}/{len(folders)}] 处理文件夹: {sub_folder_name}")

        # 在这个文件夹中创建对应的输出子目录
        output_subfolder = os.path.join(output_path, sub_folder_name + "_degraded")
        os.makedirs(output_subfolder, exist_ok=True)

        # 加载当前文件夹的动作数据
        data_dict = NpzMotion.load_data(sub_folder, min_len=1, motion_rep=motion_rep)
        data = NpzMotion(
            data=data_dict,
            motion_rep=motion_rep,
            fix_len=600,
            use_vel=cmd_args.use_vel,
        )
        file_names = data_dict["file_name"]

        print(f"  文件夹中文件数量: {len(file_names)}")

        # 处理每个动作文件
        for file_idx, fname in enumerate(file_names):
            total_files += 1
            output_fname = f"{sub_folder_name}_{fname}"
            npz_path = os.path.join(output_subfolder, output_fname)

            # 检查文件是否已存在
            # if Path(npz_path).exists() and not cmd_args.overwrite:
            #     print(
            #         f"  [{file_idx+1}/{len(file_names)}] 文件已存在，跳过: {output_fname}"
            #     )
            #     skipped_files += 1
            #     continue

            print(f"  [{file_idx+1}/{len(file_names)}] 处理文件: {fname}")

            # 获取动作数据
            motion, length, cond = data.__getitem__(index=file_idx)
            beta = data_dict["beta"][file_idx]
            length = torch.tensor([length])
            motion = motion[:length]

            # 转换为设备张量
            motion = motion_rep.normalization(motion.to(device))
            origin_motion = motion.unsqueeze(0)  # 添加batch维度

            # 解码获取pose和trans
            pose, joint, trans = motion_rep.decode(origin_motion[0])

            # 打印调试信息：查看原始维度
            # print(f"    原始pose维度: {pose.shape}")
            # print(f"    原始trans维度: {trans.shape}")

            # 应用随机退化
            # 注意：apply_random_degradations期望的输入维度为
            # pose: [batch_size, seq_len, n_joints, rotation_dim]
            # trans: [batch_size, seq_len, 3]

            # 添加batch维度

            pose = pose.unsqueeze(0)  # -> [1, seq_len, n_joints, rotation_dim]
            trans = trans.unsqueeze(0)  # -> [1, seq_len, 3]
            joint = joint.unsqueeze(0)

            # print(f"    添加batch后的pose维度: {pose.shape}")
            # print(f"    添加batch后的trans维度: {trans.shape}")

            # 应用随机退化
            degraded_pose, degraded_joint, degraded_trans = (
                degradation.apply_random_degradations(
                    pose, joint, trans, global_pose=False
                )
            )

            # print(f"    退化后的pose维度: {degraded_pose.shape}")
            # print(f"    退化后的trans维度: {degraded_trans.shape}")

            # 移除batch维度以便后续处理
            if degraded_pose.shape[0] == 1:
                degraded_pose = degraded_pose[0]
            if degraded_trans.shape[0] == 1:
                degraded_trans = degraded_trans[0]

            # print(f"    移除batch后的pose维度: {degraded_pose.shape}")
            # print(f"    移除batch后的trans维度: {degraded_trans.shape}")

            # 编码回动作表示
            # degraded_motion = motion_rep.encode(degraded_pose, degraded_trans)

            # 转换为SMPL参数并保存
            out_npz = mp.motion2npz_dict(
                pose=degraded_pose, trans=degraded_trans, frame_rate=30, betas=beta
            )

            print(f"    保存退化后的动作到: {npz_path}")
            np.savez(npz_path, **out_npz)

            processed_files += 1

    # 打印总结信息
    print("\n处理完成！")
    print(f"总文件数: {total_files}")
    print(f"处理文件数: {processed_files}")
    print(f"跳过文件数: {skipped_files}")
    print(f"输出目录: {output_path}")


if __name__ == "__main__":
    main()
