#!/usr/bin/env python3
"""
MoGenDIT CJGame MB 修复测评脚本

对 data/lightai_data/CJGame_MB/npz_split/ 下的修复数据进行：
1. 使用 MoGenDIT 的 denoise 和 ada_denoise 模式进行修复
2. 跳过修复前后长度不一致的 case
3. 对每个 case 进行质量检查器测评
4. 生成可视化报告

Usage:
    python scripts/mogendit_cjgame_eval.py --max-samples 10 --mode denoise
    python scripts/mogendit_cjgame_eval.py --mode ada_denoise --denoise-step 15
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
import numpy as np

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
    parser = argparse.ArgumentParser(description='MoGenDIT CJGame MB 修复测评')

    # 输入输出
    parser.add_argument('--input-dir', type=str,
                       default='data/lightai_data/CJGame_MB/npz_split/',
                       help='输入数据目录')
    parser.add_argument('--output-dir', type=str,
                       default='work_dirs/mogendit_cjgame_eval/',
                       help='输出目录')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='最大处理样本数')

    # 模型配置
    parser.add_argument('--model-name', type=str, default='MoreDiff-0.1B',
                       help='MoGenDIT 模型名称')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='设备')

    # 修复配置
    parser.add_argument('--mode', type=str, default='denoise',
                       choices=['denoise', 'ada_denoise'],
                       help='修复模式')
    parser.add_argument('--denoise-step', type=int, default=10,
                       help='去噪步数')

    # 测评配置
    parser.add_argument('--skip-length-mismatch', action='store_true', default=True,
                       help='跳过长度不匹配的样本')

    return parser.parse_args()


def check_length_match(original_path, cleaned_path):
    """检查修复前后长度是否匹配"""
    try:
        original_data = np.load(original_path, allow_pickle=True)
        cleaned_data = np.load(cleaned_path, allow_pickle=True)

        original_frames = original_data['poses'].shape[0]
        cleaned_frames = cleaned_data['poses'].shape[0]

        return original_frames == cleaned_frames
    except Exception as e:
        logger.warning(f'检查长度匹配失败: {e}')
        return False


def collect_evaluation_pairs(input_dir):
    """收集需要测评的修复对"""
    input_path = Path(input_dir)

    # 查找所有 *_cleaned.npz 文件
    cleaned_files = sorted(input_path.rglob('*_cleaned.npz'))

    pairs = []
    for cleaned_file in cleaned_files:
        # 找到对应的原始文件
        original_name = cleaned_file.name.replace('_cleaned.npz', '.npz')
        original_file = cleaned_file.parent / original_name

        if original_file.exists():
            pairs.append({
                'original': str(original_file),
                'cleaned': str(cleaned_file),
                'name': original_file.stem
            })

    return pairs


def run_quality_check(motion_data):
    """运行质量检查器"""
    try:
        from hftrainer.evaluation.quality_check_rules.motion_quality_checker import MotionQualityChecker

        checker = MotionQualityChecker()

        # 构建检查器输入格式
        check_input = {
            'poses': motion_data['poses'],
            'trans': motion_data['trans'],
            'betas': motion_data.get('betas', np.zeros(16)),
            'gender': motion_data.get('gender', 'neutral'),
            'mocap_framerate': motion_data.get('mocap_framerate', 30.0)
        }

        result = checker.check(check_input)

        # 转换 AggregatedCheckResult 为字典格式
        if hasattr(result, 'is_valid'):
            return {
                'is_valid': result.is_valid,
                'severity': getattr(result, 'severity', 'unknown'),
                'failed_checks': getattr(result, 'failed_checks', []),
                'borderline_checks': getattr(result, 'borderline_checks', []),
                'invalid_mask': getattr(result, 'invalid_mask', None)
            }
        else:
            # 已经是字典格式
            return result
    except Exception as e:
        logger.error(f'质量检查失败: {e}')
        return None


def create_visualization_report(pair_info, mogendit_results, quality_results, output_dir):
    """创建可视化报告"""
    report = {
        'pair_info': pair_info,
        'timestamp': time.time(),
        'mogendit_results': {},
        'quality_results': {},
        'summary': {}
    }

    # 汇总质量检查结果
    for mode, result in quality_results.items():
        if result:
            report['quality_results'][mode] = {
                'is_valid': result.get('is_valid', False),
                'severity': result.get('severity', 'unknown'),
                'failed_checks': result.get('failed_checks', []),
                'borderline_checks': result.get('borderline_checks', []),
                'invalid_mask_shape': result.get('invalid_mask', np.zeros((1, 1))).shape if result.get('invalid_mask') is not None else None
            }

    # 计算修复成功率
    valid_count = sum(1 for r in quality_results.values() if r and r.get('is_valid', False))
    total_count = len([r for r in quality_results.values() if r is not None])

    report['summary'] = {
        'total_modes': total_count,
        'valid_modes': valid_count,
        'success_rate': valid_count / total_count if total_count > 0 else 0,
        'best_mode': max(quality_results.items(), key=lambda x: x[1].get('is_valid', False) if x[1] else False)[0] if quality_results else None
    }

    # 保存报告
    report_file = Path(output_dir) / f"{pair_info['name']}_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    return report


def main():
    args = parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集测评对
    pairs = collect_evaluation_pairs(args.input_dir)
    if args.max_samples:
        pairs = pairs[:args.max_samples]

    logger.info(f'找到 {len(pairs)} 个修复对')

    # 初始化 MoGenDIT pipeline
    try:
        from hftrainer.pipelines.motion.mogendit_pipeline import MoGenDITRepairPipeline
        pipeline = MoGenDITRepairPipeline(
            model_name=args.model_name,
            device=args.device,
            use_ema=True
        )
        logger.info('MoGenDIT pipeline 初始化成功')
    except Exception as e:
        logger.error(f'MoGenDIT pipeline 初始化失败: {e}')
        return

    # 处理每个修复对
    results = []
    for i, pair in enumerate(pairs):
        logger.info(f'[{i+1}/{len(pairs)}] 处理: {pair["name"]}')

        # 检查长度匹配
        if args.skip_length_mismatch:
            if not check_length_match(pair['original'], pair['cleaned']):
                logger.warning(f'跳过长度不匹配的样本: {pair["name"]}')
                continue

        try:
            # 1. 加载原始数据
            original_data = np.load(pair['original'], allow_pickle=True)

            # 2. 使用 MoGenDIT 修复
            mogendit_results = {}

            # denoise 模式
            denoise_output = output_dir / f"{pair['name']}_denoise.npz"
            pipeline.repair_npz(
                input_path=pair['original'],
                output_path=str(denoise_output),
                mode='denoise',
                step=args.denoise_step
            )
            mogendit_results['denoise'] = str(denoise_output)

            # ada_denoise 模式
            ada_output = output_dir / f"{pair['name']}_ada_denoise.npz"
            pipeline.repair_npz(
                input_path=pair['original'],
                output_path=str(ada_output),
                mode='ada_denoise',
                step=args.denoise_step
            )
            mogendit_results['ada_denoise'] = str(ada_output)

            # 3. 质量检查
            quality_results = {}

            # 检查原始数据
            quality_results['original'] = run_quality_check(original_data)

            # 检查人工修复数据
            cleaned_data = np.load(pair['cleaned'], allow_pickle=True)
            quality_results['cleaned'] = run_quality_check(cleaned_data)

            # 检查 MoGenDIT 修复结果
            for mode, output_path in mogendit_results.items():
                mogendit_data = np.load(output_path, allow_pickle=True)
                quality_results[mode] = run_quality_check(mogendit_data)

            # 4. 生成报告
            report = create_visualization_report(pair, mogendit_results, quality_results, output_dir)

            results.append({
                'pair': pair,
                'report': report
            })

            logger.info(f'完成: {pair["name"]} - 成功率: {report["summary"]["success_rate"]:.2%}')

        except Exception as e:
            logger.error(f'处理失败 {pair["name"]}: {e}')
            continue

    # 生成总体报告
    if results:
        overall_report = {
            'total_pairs': len(results),
            'successful_pairs': len(results),
            'timestamp': time.time(),
            'config': vars(args),
            'detailed_results': [r['report'] for r in results]
        }

        overall_file = output_dir / "overall_report.json"
        with open(overall_file, 'w', encoding='utf-8') as f:
            json.dump(overall_report, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f'测评完成! 总体报告: {overall_file}')
    else:
        logger.warning('没有成功处理的样本')


if __name__ == '__main__':
    main()