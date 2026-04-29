# MoreDiff - 基于扩散模型的3D动作数据修复

## 项目简介

MoreDiff（Motion Refinement in Diffusion Steps）是一个基于扩散模型的3D人体动作修复框架，提供端到端的动作数据去噪与修复功能。

**核心特性**：
- 支持多种修复模式：去噪（denoise）、自适应去噪（ada_denoise）、位移重生成（trans_regen）
- 窗口化处理长序列动作数据
- 批量处理 .npz 动作文件，保持原始目录结构

## 模型训练

```bash
# 多GPU训练（4个GPU）
torchrun --nproc_per_node=4 --master_port=29500 -m train.train_multi_GPUs
```

标准版本的训练配置文件为 `train/train_args_2.json`，模型自动保存到 `./save/ckpt` 目录。
其余配置文件为科研实验用途。

### 数据集准备

项目使用以下数据集进行训练：

1. **AMASS数据集**：高质量动作捕捉数据集
2. **Academic数据集**：学术界开源动作数据集 包含AMASS近期更新的部分数据

数据处理脚本见：https://git.woa.com/chengxuzuo/MoreDiff-Data

## 动作修复

### 基本用法

```bash
python motion_refine.py \
    --ckpt-dir ./save/ckpt \
    --model-name MoreDiff-0.1B \
    --input-dir ./data/collected_test_data \
    --output-root ./data/refined_output \
    --mode denoise \
    --denoise-step 10
```

输出目录自动构建为：`<output-root>/<input-dir-name>_<model-name>_<mode>_<step>`

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ckpt-dir` | （必需） | 模型检查点根目录 |
| `--model-name` | （必需） | 模型名称 | 现有不同训练设置下的模型可选，推荐使用MoreDiff-0.1B
| `--input-dir` | （必需） | 输入数据目录 |
| `--output-root` | （必需） | 输出根目录 |
| `--mode` | `denoise` | 修复模式：`denoise` / `ada_denoise` / `trans_regen` |
| `--denoise-step` | `10` | 去噪步数 |
| `--step` | 最新模型 | 指定模型训练步数 |
| `--use-ema` | `False` | 使用 EMA 模型 |
| `--fast-sampling` | `True` | 快速采样（仅 trans_regen 生效） |
| `--imputation-mode` | `skip_last` | 插补模式：`skip_last` / `all` / `none` |
| `--device` | `cuda:0` | 计算设备 |
| `--skip-existing` | `False` | 跳过已存在的输出文件 |

### 修复模式选择

| 模式 | 适用场景 | 推荐参数 |
|------|----------|----------|
| `denoise` | 轻微噪声、抖动修复 | `--denoise-step 10` |
| `ada_denoise` | 智能识别并修复问题区域（推荐） | `--denoise-step 10~20` |
| `trans_regen` | 位移缺失/不自然的位移修复 | `--fast-sampling` |

## 联系信息

- 维护者：chengxuzuo
- 邮箱：zuochengxu@stu.xmu.edu.cn