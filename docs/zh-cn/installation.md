# 安装说明

## 核心环境

- Python 3.10 或更高版本
- 与本机 CPU/CUDA 环境匹配的 PyTorch
- 安装 LTX-Video 这类固定源码版本的可选集成时需要 Git

在仓库根目录安装可编辑核心环境：

```bash
python -m pip install -e .
```

`pyproject.toml` 是依赖与包元数据的唯一真相来源。`requirements.txt` 只转发到
editable project，`setup.py` 只兼容旧打包工具；这两个文件不再重复维护版本范围。

## 可选依赖组

```bash
# TensorBoard 日志
python -m pip install -e ".[logging]"

# 文档构建
python -m pip install -e ".[docs]"

# 测试与 package build 检查
python -m pip install -e ".[dev]"
```

## LTX-Video 2.5

LTX 不属于核心依赖。按用途选择推理、训练或完整安装：

```bash
python -m pip install -e ".[ltx-video-inference]"
python -m pip install -e ".[ltx-video-train]"
python -m pip install -e ".[ltx-video]"
```

这些 extra 会从官方 Lightricks/LTX-2 的已审查 commit
`400fd31054597515f47125691032c04b1c3ee24e` 安装 `ltx-core`、
`ltx-pipelines` 和/或 `ltx-trainer`。固定 Git 版本是有意设计：PyPI 当前版本线
不包含适配器所需的最新 trainer/API 组合。

这些 extra 会强制 `torch>=2.8`，因为固定源码在 import 时使用
`torch.compiler.nested_compile_region`，而 PyTorch 2.7.x 没有该 API；但 extra
不会替你选择与 GPU 匹配的 CUDA PyTorch index。LTX 训练应在 Linux + NVIDIA
CUDA 环境运行，推荐先用官方源码的 `uv sync` 准备隔离运行时，再把 HFTrainer
安装进去。gated 模型访问、许可证、硬件规划和完整命令见
[LTX-Video 2.5 指南](models/ltx_video_2_5.md)。

## Console 命令

editable 或 wheel 安装后提供：

```text
hftrainer-train
hftrainer-infer
hftrainer-ltx-infer
hftrainer-ltx-preprocess
```

源码工作流仍可直接运行相应的 `python tools/...` 入口。

## Demo 资源

下载小型内置 demo 使用的 checkpoint：

```bash
bash tools/download_checkpoints.sh
```

下载或准备 demo 数据：

```bash
python tools/download_demo_data.py --task all
```

LTX-2.5 权重需要 gated 授权，因此不会由上述 demo helper 自动下载。请先接受
模型许可证和访问条款，再按 LTX 专页手动下载。
