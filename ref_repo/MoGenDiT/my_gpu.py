import torch
import argparse
import time
import subprocess
import re
from typing import Tuple

# python my_gpu.py --gpu 0 --compute-type conv --size 128 --duration 500000


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PyTorch GPU计算负载生成脚本")
    parser.add_argument("--gpu", type=int, default=0, help="指定GPU编号（默认0）")
    parser.add_argument(
        "--duration", type=int, default=60, help="持续计算时间（秒，默认60）"
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="matrix",
        choices=["matrix", "conv", "mixed"],
        help="计算类型：matrix(矩阵乘法)、conv(卷积)、mixed(混合，默认matrix)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=4096,
        help="计算规模（矩阵大小/卷积输入尺寸，默认4096）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="批处理大小（卷积计算用，默认32）"
    )
    return parser.parse_args()


def get_gpu_utilization(gpu_id: int) -> Tuple[int, float]:
    """获取指定GPU的利用率(%)和显存占用(GB)"""
    try:
        # 调用nvidia-smi获取GPU信息
        result = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu=utilization.gpu,memory.used",
                "--format=csv,nounits,noheader",
                f"--id={gpu_id}",
            ],
            encoding="utf-8",
        ).strip()

        util, mem_used = result.split(",")
        return int(util.strip()), float(mem_used.strip()) / 1024
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
        return 0, 0.0


def matrix_compute_loop(device: torch.device, size: int):
    """矩阵乘法计算循环（高计算密度）"""
    # 创建大矩阵（float32以平衡计算和显存）
    a = torch.randn(size, size, dtype=torch.float32, device=device)
    b = torch.randn(size, size, dtype=torch.float32, device=device)

    # 预热计算
    c = torch.matmul(a, b)
    torch.cuda.synchronize()

    while True:
        time.sleep(50)
        # 连续矩阵乘法+激活操作
        c = torch.matmul(c, a) + torch.matmul(c, b)
        c = torch.relu(c)
        # 强制同步确保计算完成
        torch.cuda.synchronize()


def conv_compute_loop(device: torch.device, size: int, batch_size: int):
    """卷积计算循环（模拟CNN推理/训练）"""
    # 创建卷积层和输入数据
    conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3).to(device)
    conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1).to(device)
    conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1).to(device)
    relu = torch.nn.ReLU().to(device)
    pool = torch.nn.MaxPool2d(2).to(device)

    # 创建输入张量 (batch, channels, height, width)
    x = torch.randn(batch_size, 3, size, size, dtype=torch.float32, device=device)

    # 预热
    out = pool(relu(conv3(relu(conv2(relu(conv1(x)))))))
    torch.cuda.synchronize()

    while True:
        # 模拟CNN前向传播
        time.sleep(0.0025)
        out = conv1(x)
        out = relu(out)
        out = conv2(out)
        out = relu(out)
        out = conv3(out)
        out = relu(out)
        out = pool(out)
        # 增加计算量：多次运算
        out = out * out + out / 2
        torch.cuda.synchronize()


def mixed_compute_loop(device: torch.device, size: int, batch_size: int):
    """混合计算循环（矩阵+卷积）"""
    # 矩阵部分
    a = torch.randn(size // 2, size // 2, dtype=torch.float32, device=device)
    b = torch.randn(size // 2, size // 2, dtype=torch.float32, device=device)

    # 卷积部分
    conv = torch.nn.Conv2d(3, 64, 3).to(device)
    x = torch.randn(batch_size, 3, 256, 256, dtype=torch.float32, device=device)

    # 预热
    c = torch.matmul(a, b)
    out = conv(x)
    torch.cuda.synchronize()

    while True:
        # 交替进行矩阵和卷积计算
        time.sleep(50)
        c = torch.matmul(c, a) + torch.matmul(c, b)
        out = conv(out) + conv(x)
        c = c + out.mean()
        torch.cuda.synchronize()


def main():
    args = parse_args()

    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("错误：CUDA不可用，无法进行GPU计算")
        return

    # 设置GPU设备
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    print(f"使用GPU {args.gpu}: {torch.cuda.get_device_name(device)}")
    print(
        f"计算类型: {args.compute_type}, 规模: {args.size}, 持续时间: {args.duration}秒"
    )
    print("-" * 50)

    # 启动计算线程（后台持续计算）
    import threading

    compute_thread = threading.Thread(
        target={
            "matrix": lambda: matrix_compute_loop(device, args.size),
            "conv": lambda: conv_compute_loop(device, args.size, args.batch_size),
            "mixed": lambda: mixed_compute_loop(device, args.size, args.batch_size),
        }[args.compute_type],
        daemon=True,
    )
    compute_thread.start()

    # 实时监控GPU状态
    start_time = time.time()
    try:
        while time.time() - start_time < args.duration:
            util, mem = get_gpu_utilization(args.gpu)
            elapsed = time.time() - start_time
            print(
                f"时间: {elapsed:.1f}s | GPU利用率: {util}% | 显存占用: {mem:.2f}GB",
                end="\r",
            )
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\n检测到中断，停止计算...")
    finally:
        # 清理GPU缓存
        torch.cuda.empty_cache()
        print(f"\nGPU缓存已清理，最终状态: 利用率 {util}% | 显存 {mem:.2f}GB")


if __name__ == "__main__":
    main()
