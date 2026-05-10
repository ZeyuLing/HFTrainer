#!/usr/bin/env python3
"""
全量修复进度监控脚本

实时监控 MoGenDIT 全量修复任务的进度，生成进度报告和统计信息
"""

import time
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

def get_total_files():
    """获取数据集总文件数"""
    import subprocess
    result = subprocess.run(
        ['find', 'data/lightai_data/CJGame_MB/npz_split/', '-name', '*.npz'],
        capture_output=True, text=True
    )
    return len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0

def get_repaired_files(output_dir):
    """获取已修复文件数"""
    path = Path(output_dir)
    if not path.exists():
        return 0

    # 统计所有修复后的NPZ文件
    repaired_files = list(path.rglob('*_denoise.npz'))
    return len(repaired_files)

def get_failed_files(output_dir):
    """获取失败文件数"""
    path = Path(output_dir)
    if not path.exists():
        return 0

    # 查找错误日志或失败标记
    error_files = list(path.rglob('*_error.log'))
    return len(error_files)

def calculate_progress(total, repaired, failed):
    """计算进度百分比"""
    if total == 0:
        return 0.0
    processed = repaired + failed
    return (processed / total) * 100

def estimate_time_remaining(start_time, processed, total):
    """估算剩余时间"""
    if processed == 0:
        return "未知"

    elapsed = time.time() - start_time
    time_per_file = elapsed / processed
    remaining_files = total - processed
    remaining_time = remaining_files * time_per_file

    # 转换为可读格式
    if remaining_time < 60:
        return f"{int(remaining_time)}秒"
    elif remaining_time < 3600:
        return f"{int(remaining_time/60)}分钟"
    else:
        return f"{int(remaining_time/3600)}小时{int((remaining_time%3600)/60)}分钟"

def generate_progress_report(output_dir, start_time):
    """生成进度报告"""
    total_files = get_total_files()
    repaired_files = get_repaired_files(output_dir)
    failed_files = get_failed_files(output_dir)

    progress = calculate_progress(total_files, repaired_files, failed_files)
    processed_files = repaired_files + failed_files

    report = {
        "timestamp": time.time(),
        "total_files": total_files,
        "repaired_files": repaired_files,
        "failed_files": failed_files,
        "processed_files": processed_files,
        "progress_percentage": round(progress, 2),
        "elapsed_time_seconds": int(time.time() - start_time),
        "estimated_remaining_time": estimate_time_remaining(start_time, processed_files, total_files),
        "success_rate": round((repaired_files / max(processed_files, 1)) * 100, 2) if processed_files > 0 else 0.0
    }

    return report

def save_progress_report(report, output_dir):
    """保存进度报告"""
    report_file = Path(output_dir) / "progress_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

def display_progress_bar(percentage, width=50):
    """显示进度条"""
    filled = int(width * percentage / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {percentage:.1f}%"

def main():
    output_dir = "work_dirs/mogendit_cjgame_full_eval"
    start_time = time.time()

    logger.info("开始监控全量修复进度...")
    logger.info(f"输出目录: {output_dir}")

    while True:
        try:
            report = generate_progress_report(output_dir, start_time)

            # 显示进度信息
            print("\n" + "="*60)
            print("MoGenDIT 全量修复进度监控")
            print("="*60)
            print(f"总文件数: {report['total_files']}")
            print(f"已修复: {report['repaired_files']}")
            print(f"失败: {report['failed_files']}")
            print(f"已处理: {report['processed_files']}")
            print(f"成功率: {report['success_rate']}%")
            print(f"进度: {display_progress_bar(report['progress_percentage'])}")
            print(f"已运行: {report['elapsed_time_seconds']}秒")
            print(f"预计剩余时间: {report['estimated_remaining_time']}")
            print("="*60)

            # 保存报告
            save_progress_report(report, output_dir)

            # 检查是否完成
            if report['progress_percentage'] >= 100:
                logger.info("全量修复任务完成!")
                break

            # 每30秒更新一次
            time.sleep(30)

        except KeyboardInterrupt:
            logger.info("监控被用户中断")
            break
        except Exception as e:
            logger.error(f"监控出错: {e}")
            time.sleep(60)  # 出错后等待1分钟再重试

if __name__ == "__main__":
    main()