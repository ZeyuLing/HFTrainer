#!/usr/bin/env python3
"""
实时进度监控器 - 全量修复任务监控

实时显示修复进度、速度、预计完成时间
"""

import time
import json
import os
import sys
from pathlib import Path
import threading
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def get_total_files():
    """获取需要修复的总文件数"""
    result = subprocess.run([
        'find', 'data/lightai_data/CJGame_MB/npz_split/', '-name', '*.npz'
    ], capture_output=True, text=True)

    files = result.stdout.strip().split('\n') if result.stdout.strip() else []
    # 排除已清理的GT文件
    files_to_repair = [f for f in files if '_cleaned.npz' not in f]
    return len(files_to_repair)

def get_repaired_files(output_dir):
    """获取已修复文件数"""
    path = Path(output_dir)
    if not path.exists():
        return 0

    # 统计所有修复后的NPZ文件
    repaired_files = list(path.rglob('*_denoise.npz'))
    return len(repaired_files)

def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{int(seconds)}秒"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes}分钟"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}小时{minutes}分钟"

def display_progress_bar(percentage, width=50):
    """显示进度条"""
    filled = int(width * percentage / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {percentage:.1f}%"

def monitor_progress():
    """监控进度主函数"""
    output_dir = "work_dirs/mogendit_multigpu_full_repair"
    start_time = time.time()

    print("🚀 MoGenDIT 多GPU全量修复进度监控")
    print("=" * 70)

    total_files = get_total_files()
    print(f"📊 总文件数: {total_files}")
    print(f"🖥️  使用GPU: 8个并行")
    print(f"⏰ 开始时间: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    print("-" * 70)

    last_count = 0
    last_time = start_time

    while True:
        try:
            repaired_files = get_repaired_files(output_dir)
            current_time = time.time()
            elapsed = current_time - start_time

            # 计算进度
            progress = (repaired_files / total_files) * 100 if total_files > 0 else 0

            # 计算速度
            time_diff = current_time - last_time
            if time_diff >= 30:  # 每30秒更新一次速度
                files_diff = repaired_files - last_count
                speed = files_diff / time_diff if time_diff > 0 else 0
                last_count = repaired_files
                last_time = current_time

            # 估算剩余时间
            if repaired_files > 0:
                time_per_file = elapsed / repaired_files
                remaining_files = total_files - repaired_files
                eta = time_per_file * remaining_files
            else:
                eta = 0

            # 清屏并显示进度
            os.system('clear' if os.name == 'posix' else 'cls')

            print("🚀 MoGenDIT 多GPU全量修复进度监控")
            print("=" * 70)
            print(f"📊 总文件数: {total_files}")
            print(f"✅ 已修复: {repaired_files}")
            print(f"📈 进度: {display_progress_bar(progress)}")
            print(f"⚡ 修复速度: {speed:.1f} 文件/秒")
            print(f"⏱️  已运行: {format_time(elapsed)}")
            print(f"⏳ 预计剩余: {format_time(eta)}")
            print(f"🕐 预计完成: {time.strftime('%H:%M:%S', time.localtime(start_time + elapsed + eta))}")
            print("-" * 70)

            # 显示最近修复的文件
            print("📁 最近修复的文件:")
            try:
                # 获取最新的5个修复文件
                path = Path(output_dir)
                if path.exists():
                    recent_files = sorted(path.rglob('*_denoise.npz'), key=os.path.getmtime, reverse=True)[:5]
                    for i, file_path in enumerate(recent_files):
                        print(f"   {i+1}. {file_path.name}")
            except:
                print("   暂无修复文件")

            print("-" * 70)
            print("按 Ctrl+C 停止监控")

            # 检查是否完成
            if repaired_files >= total_files:
                print("🎉 修复任务完成!")
                break

            time.sleep(5)  # 每5秒更新一次

        except KeyboardInterrupt:
            print("\n⏹️  监控已停止")
            break
        except Exception as e:
            print(f"❌ 监控出错: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_progress()