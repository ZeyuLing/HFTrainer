# 安装说明

## 环境要求

- Python 3.9+
- PyTorch 2.0+
- `accelerate`、`transformers`、`diffusers`、`datasets`、`mmengine`
- 图像/视频 demo 依赖 `torchvision`

## 安装

```bash
pip install -e .
```

可选可视化依赖：

```bash
pip install tensorboard
```

LoRA 功能依赖 `peft`。当前已经验证的版本范围是：

```bash
pip install "accelerate>=1.1,<2" "peft>=0.17,<0.18"
```

## 下载 Demo 资源

下载 demo config 依赖的 checkpoint：

```bash
bash tools/download_checkpoints.sh
```

准备 demo 数据：

```bash
python3 tools/download_demo_data.py --task all
```

仓库中已经带有最小 demo 数据目录和本地 checkpoint 结构，首次阅读代码时可以直接参考 `data/` 与 `checkpoints/`。
