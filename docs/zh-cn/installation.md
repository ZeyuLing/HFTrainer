# 安装说明

## 核心环境

- Python 3.10 或更高版本
- 与本机 CPU/CUDA 环境匹配的 PyTorch

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

HFTrainer 已经包含经过修改并固定版本的 LTX 模型、trainer、预处理与 pipeline
源码。LTX 不进入基础依赖，是因为 22B 工作流还需要额外的媒体与科学计算支持库。
按用途选择推理、训练或完整安装：

```bash
python -m pip install -e ".[ltx-video-inference]"
python -m pip install -e ".[ltx-video-train]"
python -m pip install -e ".[ltx-video]"

# 可选：实验跟踪 / Hub 发布
python -m pip install -e ".[ltx-video-integrations]"

# 可选：EXR/HDR 媒体路径
python -m pip install -e ".[ltx-video-hdr]"
```

这些 extra 只安装 PyAV、Einops、SciPy、Pydantic、Rich、torchaudio、pandas、
Pillow-HEIF 等支持库；W&B/Hub 发布与 EXR/HDR 处理保持为独立的显式可选组。
任何一组都**不会**安装 `ltx-core`、`ltx-pipelines`、`ltx-trainer` 或其他模型
框架，也不需要第二份 LTX checkout。

extra 要求 `torch>=2.8`；请先为目标机器选择正确的 CUDA PyTorch wheel。完整
训练路径当前支持 Linux + NVIDIA CUDA。源码与许可证边界、gated 权重访问、完整
命令、验证范围和硬件规划见
[LTX-Video 2.5 指南](models/ltx_video_2_5.md)。

## 模型依赖边界

模型实现、tokenizer、采样 scheduler、LoRA、artifact loader、trainer 与 pipeline
都从 `hftrainer.*` 执行。项目仍正常依赖 PyTorch、Accelerate、MMEngine、
safetensors、NumPy、Pillow 等通用基础设施，但不要求安装外部模型实现包；是否
安装这类包也不能改变 config 最终解析到的模型代码。

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
