#!/usr/bin/env python3
"""
可视化 MoGenDIT 修复结果对比

生成修复前后动作的并排对比可视化，包含：
- 原始动作
- 人工修复 (cleaned)
- MoGenDIT denoise 修复
- MoGenDIT ada_denoise 修复

Usage:
    python scripts/visualize_mogendit_results.py --input-dir work_dirs/mogendit_cjgame_eval/
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='MoGenDIT 修复结果可视化')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='包含修复结果和报告的目录')
    parser.add_argument('--output-dir', type=str, default='work_dirs/mogendit_visualization/',
                       help='可视化输出目录')
    parser.add_argument('--max-samples', type=int, default=5,
                       help='最大可视化样本数')
    return parser.parse_args()


def load_motion_data(npz_path):
    """加载 NPZ 运动数据"""
    try:
        data = np.load(npz_path, allow_pickle=True)
        return {
            'poses': data['poses'],
            'trans': data['trans'],
            'betas': data.get('betas', np.zeros(16)),
            'gender': data.get('gender', 'neutral'),
            'fps': data.get('mocap_framerate', 30.0)
        }
    except Exception as e:
        logger.error(f'加载运动数据失败 {npz_path}: {e}')
        return None


def extract_joint_positions(motion_data):
    """从 SMPL 姿态中提取关节位置（简化版）"""
    # 简化的关节位置计算 - 实际应该使用 SMPL 前向运动学
    poses = motion_data['poses']
    trans = motion_data['trans']

    # 假设 22 个关节，每个关节 3D 位置
    T = poses.shape[0]
    joint_positions = np.zeros((T, 22, 3))

    # 简化：使用根位置 + 关节相对偏移
    for t in range(T):
        joint_positions[t, 0] = trans[t]  # 根关节
        # 其他关节的简化位置
        for j in range(1, 22):
            # 使用姿态向量的部分信息作为相对偏移
            offset = poses[t, j*3:j*3+3] * 0.1  # 缩放因子
            joint_positions[t, j] = joint_positions[t, 0] + offset

    return joint_positions


def create_comparison_animation(original_data, cleaned_data, denoise_data, ada_data, output_path):
    """创建修复结果对比动画"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MoGenDIT 修复效果对比', fontsize=16)

    # 提取关节位置
    original_joints = extract_joint_positions(original_data)
    cleaned_joints = extract_joint_positions(cleaned_data)
    denoise_joints = extract_joint_positions(denoise_data)
    ada_joints = extract_joint_positions(ada_data)

    T = min(original_joints.shape[0], cleaned_joints.shape[0],
            denoise_joints.shape[0], ada_joints.shape[0])

    # 设置子图
    titles = ['原始动作', '人工修复', 'MoGenDIT denoise', 'MoGenDIT ada_denoise']
    joints_list = [original_joints, cleaned_joints, denoise_joints, ada_joints]

    def update(frame):
        for i, ax in enumerate(axes.flat):
            ax.clear()
            ax.set_title(titles[i], fontsize=12)

            # 绘制当前帧的骨架
            joints = joints_list[i][frame]

            # 简化的骨架连接
            connections = [
                (0, 1), (0, 2), (0, 3),  # 躯干
                (1, 4), (4, 7), (7, 10),  # 左腿
                (2, 5), (5, 8), (8, 11),  # 右腿
                (3, 6), (6, 9), (9, 12),  # 脊柱
                (9, 13), (13, 14), (14, 15),  # 左臂
                (9, 16), (16, 17), (17, 18),  # 右臂
                (9, 19), (19, 20), (20, 21)   # 头部
            ]

            # 绘制关节点
            ax.scatter(joints[:, 0], joints[:, 2], c='blue', s=30)

            # 绘制骨架连接
            for start, end in connections:
                if start < len(joints) and end < len(joints):
                    ax.plot([joints[start, 0], joints[end, 0]],
                           [joints[start, 2], joints[end, 2]], 'r-')

            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

            # 显示帧数
            ax.text(0.02, 0.98, f'Frame: {frame}/{T}', transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        return axes.flat

    # 创建动画
    anim = FuncAnimation(fig, update, frames=T, interval=100, blit=False)

    # 保存动画
    anim.save(output_path, writer='pillow', fps=10)
    plt.close(fig)

    return output_path


def create_quality_comparison_chart(quality_results, output_path):
    """创建质量检查结果对比图表"""
    fig, ax = plt.subplots(figsize=(12, 8))

    modes = ['original', 'cleaned', 'denoise', 'ada_denoise']
    mode_names = ['原始', '人工修复', 'MoGenDIT denoise', 'MoGenDIT ada_denoise']

    # 提取质量指标
    valid_scores = []
    failed_checks = []
    borderline_checks = []

    for mode in modes:
        if mode in quality_results and quality_results[mode]:
            result = quality_results[mode]
            valid_scores.append(1.0 if result.get('is_valid', False) else 0.0)
            failed_checks.append(len(result.get('failed_checks', [])))
            borderline_checks.append(len(result.get('borderline_checks', [])))
        else:
            valid_scores.append(0.0)
            failed_checks.append(0)
            borderline_checks.append(0)

    # 创建条形图
    x = np.arange(len(modes))
    width = 0.25

    ax.bar(x - width, valid_scores, width, label='通过检查', color='green', alpha=0.7)
    ax.bar(x, failed_checks, width, label='失败检查', color='red', alpha=0.7)
    ax.bar(x + width, borderline_checks, width, label='边界检查', color='orange', alpha=0.7)

    ax.set_xlabel('修复模式')
    ax.set_ylabel('检查结果数量')
    ax.set_title('质量检查结果对比')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_names)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 添加数值标签
    for i, (v, f, b) in enumerate(zip(valid_scores, failed_checks, borderline_checks)):
        ax.text(i - width, v + 0.05, f'{v:.1f}', ha='center', va='bottom')
        ax.text(i, f + 0.05, str(f), ha='center', va='bottom')
        ax.text(i + width, b + 0.05, str(b), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def main():
    args = parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载总体报告
    report_path = Path(args.input_dir) / 'overall_report.json'
    if not report_path.exists():
        logger.error(f'报告文件不存在: {report_path}')
        return

    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    logger.info(f'加载报告: {len(report["detailed_results"])} 个样本')

    # 处理每个样本
    for i, result in enumerate(report['detailed_results'][:args.max_samples]):
        pair_info = result['pair_info']
        name = pair_info['name']

        logger.info(f'[{i+1}/{min(len(report["detailed_results"]), args.max_samples)}] 处理: {name}')

        # 加载所有版本的运动数据
        data_files = {
            'original': pair_info['original'],
            'cleaned': pair_info['cleaned'],
            'denoise': Path(args.input_dir) / f'{name}_denoise.npz',
            'ada_denoise': Path(args.input_dir) / f'{name}_ada_denoise.npz'
        }

        motion_data = {}
        for key, file_path in data_files.items():
            if Path(file_path).exists():
                motion_data[key] = load_motion_data(file_path)
            else:
                logger.warning(f'文件不存在: {file_path}')
                motion_data[key] = None

        # 检查是否所有数据都加载成功
        if all(data is not None for data in motion_data.values()):
            # 创建对比动画
            anim_path = output_dir / f'{name}_comparison.gif'
            create_comparison_animation(
                motion_data['original'],
                motion_data['cleaned'],
                motion_data['denoise'],
                motion_data['ada_denoise'],
                anim_path
            )
            logger.info(f'创建动画: {anim_path}')

            # 创建质量对比图表
            if 'quality_results' in result:
                chart_path = output_dir / f'{name}_quality_chart.png'
                create_quality_comparison_chart(result['quality_results'], chart_path)
                logger.info(f'创建质量图表: {chart_path}')
        else:
            logger.warning(f'数据不完整，跳过可视化: {name}')

    logger.info(f'可视化完成! 输出目录: {output_dir}')


if __name__ == '__main__':
    main()