# LTX-Video 2.5

HFTrainer 通过一层轻量、配置驱动的适配器接入 Lightricks 官方的
`ltx-core`、`ltx-pipelines` 和 `ltx-trainer`。模型加载、去噪、数据预处理、
优化器与 checkpoint、音视频编码仍由官方实现负责；HFTrainer 提供统一的
registry/config 接口、严格的分体权重校验和可复现的命令入口。

当前适配器固定到 Lightricks/LTX-2 commit
`400fd31054597515f47125691032c04b1c3ee24e`。升级时不要单独更新三个 LTX
包中的某一个，它们的内部 API 是一起演进的。

官方资料：

- [LTX-2 源码](https://github.com/Lightricks/LTX-2)
- [LTX-2.5 模型与权重](https://huggingface.co/Lightricks/LTX-2.5)
- [LTX Trainer 快速开始](https://docs.ltx.io/open-source-model/ltx-trainer/quick-start)
- [LTX-2.5 许可证](https://github.com/Lightricks/LTX-2/blob/400fd31054597515f47125691032c04b1c3ee24e/LICENSE)

## 支持范围与验证边界

| 能力 | HFTrainer 接口 | 本仓库实际完成的验证 |
| --- | --- | --- |
| 蒸馏版文/图生同步音视频 | `LTXVideoBundle` + `LTXVideoPipeline` | config 解析、权重角色校验、registry 构建、参数映射和 mock 官方 API 合约 |
| Dev 两阶段 + LoRA 推理 | 同一 bundle/pipeline，设置 `mode='dev_two_stage'` | config 和官方 API 合约；校验官方 distilled LoRA 与用户 LoRA 是两份独立权重 |
| 文生同步音视频 LoRA 训练 | `LTXVideoTrainer` managed trainer | 官方 Pydantic config 映射、预处理命令、权重角色和 managed-runner 分发 |
| 22B 模型真实 GPU 生成/训练 | 在具备权重和受支持硬件时可执行 | **没有在本仓库的验证环境中实跑** |

最后一行非常重要：轻量测试通过说明适配器与固定版本的 Python API 对齐，
不代表已经验证生成质量、速度、训练收敛或某张 GPU 上的 22B 完整运行。

在兼容的官方环境中，可用 optional real-source contract 直接调用真实 Pydantic
配置类并检查公开签名：

```bash
HFTRAINER_LTX_SOURCE_ROOT=third_party/LTX-2 \
  python -m pytest -m upstream \
  tests/integration/test_ltx_video_official_contract.py
```

## 系统、CUDA 与显存规划

- HFTrainer 核心要求 Python 3.10+；固定版本的 LTX 包元数据也接受
  Python 3.10+。模型页仍写着 Python 3.12+、CUDA 12.7+ 和 PyTorch 2.7
  系列，但当前固定源码会在模块加载时直接使用
  `torch.compiler.nested_compile_region`。PyTorch 2.7.x 没有该 API，因此
  HFTrainer 会执行等价于 PyTorch 2.8+ 的能力预检，并给出明确错误，而不是
  让用户遇到上游深层 `AttributeError`。
- 训练使用 Linux + NVIDIA CUDA。官方 trainer 依赖面向 CUDA/Triton 的运行栈，
  `LTXVideoTrainer` 默认会拒绝非 Linux 训练。官方推理包在其他系统上存在
  fallback，但 HFTrainer 不把它标记为已验证的生产路径。
- 官方 trainer 对标准配置建议 80 GB 或更大显存；官方 low-VRAM 配置面向
  32 GB GPU，通过 INT8 量化、8-bit optimizer、较低 LoRA rank 和 gradient
  checkpointing 降低占用。这只是容量规划参考，不是通用保证；分辨率、帧数、
  是否生成音频、validation、VAE 选择和软件版本都会影响峰值显存。
- 下方蒸馏推理所需的官方权重下载约 66 GiB。磁盘占用不等于显存占用。
  若使用 FP8 或 CPU/disk offload，请先核对固定版本的官方文档并检查输出质量。

## 安装兼容运行时

生产环境推荐让固定版本的官方 checkout 选择相互匹配的 PyTorch/CUDA 包，再把
HFTrainer 加入该隔离环境，避免 pip 改写官方运行栈。在 Linux 的 HFTrainer
仓库根目录执行：

```bash
git clone https://github.com/Lightricks/LTX-2.git third_party/LTX-2
git -C third_party/LTX-2 checkout 400fd31054597515f47125691032c04b1c3ee24e

cd third_party/LTX-2
uv sync
uv pip install --python .venv/bin/python --no-deps -e ../..
uv pip install --python .venv/bin/python "mmengine>=0.7,<1" "PyYAML>=6"
cd ../..
```

这样会保留官方的 runtime/index 决策，并与其他 HFTrainer 项目隔离。这里
`--no-deps` 是有意的：官方环境已经负责 PyTorch、Accelerate、Transformers 和
LTX 包，只需补充 HFTrainer 的轻量 config 依赖。

如果已经准备好与 CUDA 匹配、且包含
`torch.compiler.nested_compile_region` 的 PyTorch 环境，可以使用较短的 extra
安装路径。按用途选择：

```bash
python -m pip install -e ".[ltx-video-inference]"
python -m pip install -e ".[ltx-video-train]"

# 同时安装训练与推理
python -m pip install -e ".[ltx-video]"
```

这些 extra 会从同一个 Git commit 安装全部 LTX 组件，不会误用 PyPI 上较旧且
API 不兼容的版本线。extra 要求 `torch>=2.8`，但 Python package extra 无法替
每台机器选择正确的 CUDA wheel/index，不能用这条便利命令代替 CUDA 运行时规划。

## 接受许可证并下载 gated 权重

LTX-2.5 是 Hugging Face gated 仓库。先打开
[模型页面](https://huggingface.co/Lightricks/LTX-2.5)，阅读并接受访问条款，
再登录一个具有 gated repository 读取权限的 token：

```bash
hf auth login
```

模型采用 **LTX-2.x Community License**，不是 Apache/MIT 一类开源许可证。
官方模型页目前说明：主体年收入低于 1000 万美元时可免费商业/生产使用，超过该
阈值需要付费协议；转让 fine-tune 权重也可能需要付费许可。商业使用或分发权重前
必须阅读完整许可证，模型页摘要不能替代具有约束力的法律文本。

统一设置 checkpoint 根目录：

```bash
export LTX25_CHECKPOINT_ROOT="$PWD/checkpoints/LTX-2.5"
```

蒸馏推理示例需要以下分体权重：

```bash
hf download Lightricks/LTX-2.5 \
  diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  vae/ltx-2.5-video-vae-bf16.safetensors \
  vae/ltx-2.5-audio-vae-bf16.safetensors \
  model_patches/ltx-2.5-duration-head-bf16.safetensors \
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --local-dir "$LTX25_CHECKPOINT_ROOT"
```

训练和 Dev + LoRA 引导推理还需要：

```bash
hf download Lightricks/LTX-2.5 \
  diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors \
  --local-dir "$LTX25_CHECKPOINT_ROOT"
```

若 `hf download` 返回 401/403，请同时检查模型页是否已经接受条款，以及 token
是否具备 gated repository read 权限。

## 各权重的角色

| 组件 | 用途 | 关键约束 |
| --- | --- | --- |
| distilled transformer | 快速推理基座 | 使用官方固定蒸馏 schedule，不能作为训练基座 |
| dev transformer | 可训练的完整 DiT | LoRA/full training 和引导式两阶段推理必需 |
| 带投影的 Gemma 4 text encoder | 文本特征与 LTX projection | 不能替换成 Google 原生 Gemma 4 |
| video VAE | 视频 latent 编解码 | 示例使用质量更高、也更重的 DiffVAE BF16 权重 |
| audio VAE/vocoder | 同步音频编解码 | 当前联合音视频训练配置必需 |
| spatial upsampler | 第二阶段 2x latent 上采样 | 两种示例推理模式都需要 |
| duration head | 可选的 prompt 时长预测 | 示例 config 填了该路径，因此要求文件存在 |
| 官方 distilled LoRA | Dev 两阶段流程的阶段衔接 | 与用户训练 LoRA 同时存在，不能互相替代 |
| 用户 LoRA | 训练得到的任务/风格适配 | 用 `LTX25_USER_LORA` 传入实际保存的 `.safetensors` 路径 |

不要把 `*-comfy-int8-convrot.safetensors` 传给原生 PyTorch pipeline；这些文件
用于 ComfyUI。HFTrainer 会在加载 22B 模型前拒绝此类错误路径。

## 蒸馏版快速推理

该路径调用官方 `DistilledPipeline`。它使用固定去噪 schedule；由于 checkpoint
采用 CFG=1 和官方预定义 sigma，HFTrainer 会主动拒绝
`--num-inference-steps` 和 negative prompt。

```bash
hftrainer-ltx-infer \
  configs/ltx_video/infer_ltx_video_2_5_distilled.py \
  --prompt "A paper boat drifts through a rain-filled street at dusk." \
  --output outputs/ltx_video_2_5/distilled.mp4
```

源码目录下的等价入口：

```bash
python tools/infer_ltx_video.py \
  configs/ltx_video/infer_ltx_video_2_5_distilled.py \
  --prompt "A paper boat drifts through a rain-filled street at dusk." \
  --output outputs/ltx_video_2_5/distilled.mp4
```

可追加 `--image path/to/frame.jpg` 做首帧条件。配置 duration-head 权重后也可使用
`--auto-duration`，它与 `--num-frames` 互斥。

HFTrainer 当前的两阶段适配器要求
高宽都能被 64 整除，并满足 `num_frames % 8 == 1`；示例为 768x512、121 帧。
官方基础模型/VAE 的底层约束是 32 对齐；HFTrainer 当前两阶段最终输出路径会在
加载权重前主动应用更严格的 64 对齐。

## 准备训练数据

官方预处理器接受 CSV、JSON 或 JSONL，最小字段为 `caption` 与 `video`：

```json
[
  {
    "caption": "A handheld shot follows a cyclist through a quiet alley.",
    "video": "videos/cyclist.mp4"
  }
]
```

预处理实际调用固定版本的官方脚本。由于该脚本没有作为稳定 console entry
point 发布，需要保留一份同 commit 的源码：

```bash
git clone https://github.com/Lightricks/LTX-2.git third_party/LTX-2
git -C third_party/LTX-2 checkout 400fd31054597515f47125691032c04b1c3ee24e
```

计算视频/音频 latent 与 Gemma 特征：

```bash
hftrainer-ltx-preprocess data/ltx_video_2_5/dataset.json \
  --ltx-repo third_party/LTX-2 \
  --resolution-buckets 960x544x49 \
  --model-path "$LTX25_CHECKPOINT_ROOT/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
  --text-encoder-path "$LTX25_CHECKPOINT_ROOT/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --video-vae-path "$LTX25_CHECKPOINT_ROOT/vae/ltx-2.5-video-vae-bf16.safetensors" \
  --audio-vae-path "$LTX25_CHECKPOINT_ROOT/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --output-dir data/ltx_video_2_5/.precomputed
```

只有当训练策略也不生成音频时才使用 `--skip-audio`。切换 checkpoint 版本、
resolution bucket 或 text encoder 后要加 `--overwrite` 重新处理；LTX-2.3 与
LTX-2.5 的缓存文本特征不能混用。

## 训练 LoRA

示例 config 在 `trainer.native_config` 下完整表达官方 `LtxTrainerConfig` schema。
设置权重与数据目录后，从统一训练入口启动：

```bash
export LTX25_PREPROCESSED_DATA="$PWD/data/ltx_video_2_5/.precomputed"
export HFTRAINER_WORK_DIR="$PWD/outputs/training/ltx_video_2_5_lora"

hftrainer-train configs/ltx_video/train_ltx_video_2_5_lora.py
```

源码目录下多卡启动：

```bash
accelerate launch tools/train.py configs/ltx_video/train_ltx_video_2_5_lora.py
```

`LTXVideoTrainer` 是 managed trainer：它不会把官方算法重新翻译成 HFTrainer
的 `train_step`。适配器先校验并保存 resolved config，再由官方 `LtxvTrainer`
完整负责 Accelerator、optimizer、checkpoint、validation 和 resume 语义，避免
维护第二套逐渐偏离上游的训练实现。

仓库配置是起始 recipe，不代表最优超参数或 benchmark 复现。若使用 32 GB 级别
GPU，请参考官方
[low-VRAM 配置](https://github.com/Lightricks/LTX-2/blob/400fd31054597515f47125691032c04b1c3ee24e/packages/ltx-trainer/configs/t2v_lora_low_vram.yaml)
迁移显存优化项，并针对自己的数据验证质量。

## 使用 Dev 模型和训练 LoRA 推理

将 `LTX25_USER_LORA` 指向 trainer 实际生成的 LoRA 文件：

```bash
export LTX25_USER_LORA="$PWD/outputs/training/ltx_video_2_5_lora/checkpoints/<saved-lora>.safetensors"

hftrainer-ltx-infer \
  configs/ltx_video/infer_ltx_video_2_5_dev_lora.py \
  --prompt "A slow dolly shot moves through an art studio while rain taps the windows." \
  --negative-prompt "blurry, distorted, low quality, artifacts" \
  --num-inference-steps 30 \
  --output outputs/ltx_video_2_5/dev_lora.mp4
```

这一路径同时使用 dev transformer、两阶段流程所需的官方 distilled LoRA，以及
用户训练 LoRA。与 distilled transformer 快速路径不同，它支持 guidance、
negative prompt 和可配置步数。不要为了更快而把用户 LoRA 直接塞进蒸馏版 config；
两条路径的模型合约不同。

## Config 注册与轻量 import

LTX 依赖是可选的，只有真正构建 LTX backend 时才加载。每份 config 用
`custom_imports` 显式注册所需纵向模块：

```python
custom_imports = dict(
    imports=[
        'hftrainer.models.ltx_video',
        'hftrainer.pipelines.ltx_video',
    ],
    allow_failed_imports=False,
)
```

训练 config 则导入 `hftrainer.trainers.ltx_video`。单纯 `import hftrainer` 不会
加载 Transformers、Diffusers、Accelerate 或 LTX。只有确实需要全部内置模块的
应用才调用 `hftrainer.register_all_modules()`；普通配置应优先使用精确的
`custom_imports`。

## 常见问题

- **缺少 `ltx_*` module：**安装对应 extra；同一环境训练和推理时安装
  `ltx-video`。
- **下载返回 401/403：**接受 gated 模型条款，并使用有 gated repo read 权限的
  token。
- **权重角色错误：**训练必须使用 `dev-transformer`，蒸馏推理必须使用
  `distilled-transformer`，text encoder 文件名应包含
  `gemma4-12b-with-proj-ltx-2.5`。
- **CUDA OOM：**降低分辨率/帧数 bucket、保持 batch size 1、开启 gradient
  checkpointing，并参考官方
  [trainer troubleshooting](https://github.com/Lightricks/LTX-2/blob/400fd31054597515f47125691032c04b1c3ee24e/packages/ltx-trainer/docs/troubleshooting.md)。
- **config 测试通过但生成失败：**合约测试不会分配 22B 权重。报告问题时记录 GPU、
  driver、CUDA/PyTorch 版本、准确 commit、完整命令和第一段异常堆栈。
