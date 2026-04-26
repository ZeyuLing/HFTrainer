#!/usr/bin/env python3
"""
MoGenDIT Pipeline 多GPU并行修复脚本

使用正确的 MoGenDITRepairPipeline 进行修复，而不是测评脚本
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
import multiprocessing as mp
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

def get_gpu_count():
    """获取可用GPU数量"""
    try:
        return torch.cuda.device_count()
    except:
        return 1

def get_files_to_repair(input_dir):
    """获取需要修复的文件列表（排除已清理的GT文件）"""
    input_path = Path(input_dir)

    # 查找所有NPZ文件，但排除 _cleaned.npz
    all_npz_files = sorted(input_path.rglob('*.npz'))

    files_to_repair = []
    for npz_file in all_npz_files:
        if '_cleaned.npz' not in str(npz_file):
            files_to_repair.append(str(npz_file))

    logger.info(f'找到 {len(files_to_repair)} 个需要修复的文件（排除 {len(all_npz_files) - len(files_to_repair)} 个GT文件）')
    return files_to_repair

def repair_single_file(file_path, gpu_id, output_dir, mode='denoise', step=10):
    """在指定GPU上修复单个文件"""
    try:
        # 构建输出路径
        input_path = Path(file_path)
        output_name = input_path.stem + f'_{mode}.npz'
        output_path = Path(output_dir) / output_name

        # 如果输出文件已存在，跳过
        if output_path.exists():
            logger.info(f'GPU{gpu_id}: 跳过已修复文件 {input_path.name}')
            return {'file': file_path, 'status': 'skipped', 'gpu': gpu_id}

        # 导入并初始化 MoGenDIT pipeline
        from hftrainer.pipelines.motion.mogendit_pipeline import MoGenDITRepairPipeline

        # 初始化 pipeline
        pipeline = MoGenDITRepairPipeline(
            model_name='MoreDiff-0.1B',
            device=f'cuda:{gpu_id}',
            use_ema=True
        )

        logger.info(f'GPU{gpu_id}: 开始修复 {input_path.name}')

        # 使用 pipeline 进行修复
        pipeline.repair_npz(
            input_path=str(file_path),
            output_path=str(output_path),
            mode=mode,
            step=step
        )

        logger.info(f'GPU{gpu_id}: 成功修复 {input_path.name}')
        return {'file': file_path, 'status': 'success', 'gpu': gpu_id}

    except Exception as e:
        logger.error(f'GPU{gpu_id}: 修复失败 {file_path}: {e}')
        return {'file': file_path, 'status': 'failed', 'gpu': gpu_id, 'error': str(e)}

def worker_process(gpu_id, file_queue, output_dir, mode, step, result_queue):
    """工作进程函数"""
    try:
        while True:
            # 从队列获取文件
            try:
                file_path = file_queue.get(timeout=1)
            except:
                break  # 队列为空，退出

            # 修复文件
            result = repair_single_file(file_path, gpu_id, output_dir, mode, step)
            result_queue.put(result)

    except Exception as e:
        logger.error(f'工作进程 GPU{gpu_id} 异常: {e}')

def run_parallel_repair(files, output_dir, num_gpus=4, mode='denoise', step=10):
    """并行修复主函数"""
    start_time = time.time()

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 设置多进程
    manager = mp.Manager()
    file_queue = manager.Queue()
    result_queue = manager.Queue()

    # 将文件加入队列
    for file_path in files:
        file_queue.put(file_path)

    # 启动工作进程
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, file_queue, output_dir, mode, step, result_queue)
        )
        p.start()
        processes.append(p)

    # 收集结果
    results = []
    completed = 0
    total_files = len(files)

    while completed < total_files:
        try:
            result = result_queue.get(timeout=30)
            results.append(result)
            completed += 1

            # 显示进度
            progress = (completed / total_files) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / completed) * (total_files - completed) if completed > 0 else 0

            logger.info(f'进度: {completed}/{total_files} ({progress:.1f}%) - 已运行: {elapsed:.0f}s - 预计剩余: {eta:.0f}s')

        except:
            # 检查是否所有进程都已完成
            if all(not p.is_alive() for p in processes):
                break

    # 等待所有进程结束
    for p in processes:
        p.join(timeout=10)

    # 分析结果
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')

    total_time = time.time() - start_time

    # 生成报告
    report = {
        'timestamp': time.time(),
        'total_files': total_files,
        'success_files': success_count,
        'failed_files': failed_count,
        'skipped_files': skipped_count,
        'success_rate': (success_count / total_files) * 100 if total_files > 0 else 0,
        'total_time_seconds': total_time,
        'average_time_per_file': total_time / max(success_count + failed_count, 1),
        'gpu_count': num_gpus,
        'mode': mode,
        'step': step,
        'results': results
    }

    return report

def main():
    parser = argparse.ArgumentParser(description='MoGenDIT Pipeline 多GPU并行修复')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='输入数据目录')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--num-gpus', type=int, default=None,
                       help='使用的GPU数量（默认自动检测）')
    parser.add_argument('--mode', type=str, default='denoise',
                       choices=['denoise', 'ada_denoise'],
                       help='修复模式')
    parser.add_argument('--denoise-step', type=int, default=10,
                       help='去噪步数')
    parser.add_argument('--max-files', type=int, default=None,
                       help='最大处理文件数（测试用）')

    args = parser.parse_args()

    # 确定GPU数量
    if args.num_gpus is None:
        args.num_gpus = get_gpu_count()

    logger.info(f'检测到 {args.num_gpus} 个可用GPU')

    # 获取需要修复的文件
    files = get_files_to_repair(args.input_dir)

    if args.max_files:
        files = files[:args.max_files]
        logger.info(f'限制处理前 {args.max_files} 个文件（测试模式）')

    if not files:
        logger.error('没有找到需要修复的文件')
        return

    # 运行并行修复
    logger.info(f'开始多GPU并行修复: {len(files)} 个文件')

    report = run_parallel_repair(
        files=files,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        mode=args.mode,
        step=args.denoise_step
    )

    # 保存报告
    report_file = Path(args.output_dir) / 'pipeline_multigpu_repair_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 打印总结
    logger.info('='*60)
    logger.info('MoGenDIT Pipeline 多GPU修复任务完成!')
    logger.info(f'总文件数: {report["total_files"]}')
    logger.info(f'成功修复: {report["success_files"]}')
    logger.info(f'修复失败: {report["failed_files"]}')
    logger.info(f'跳过文件: {report["skipped_files"]}')
    logger.info(f'成功率: {report["success_rate"]:.2f}%')
    logger.info(f'总耗时: {report["total_time_seconds"]:.0f}秒')
    logger.info(f'平均时间: {report["average_time_per_file"]:.2f}秒/文件')
    logger.info(f'使用GPU: {report["gpu_count"]}')
    logger.info(f'报告文件: {report_file}')
    logger.info('='*60)

if __name__ == '__main__':
    main()