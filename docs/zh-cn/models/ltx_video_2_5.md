# LTX-Video 2.5

HFTrainer 随仓库维护一份经过修改并固定 revision 的 LTX-2 源码快照，并按框架边界重新组织：

```text
hftrainer/models/ltx_video/network/             模型数学与 loader
hftrainer/pipelines/ltx_video/backend/           推理 backend
hftrainer/trainers/ltx_video/native/             训练实现
hftrainer/trainers/ltx_video/preprocess_scripts/ 数据预处理
```

运行时不需要另装 `ltx-core`、`ltx-pipelines` 或 `ltx-trainer`，也不需要第二份源码 checkout。内部 import 全部指向 HFTrainer 本地命名空间；LTX 使用的 Gemma 文本路径和 LoRA 注入也由仓库本地实现。

## 来源与许可证

- 源码：https://github.com/Lightricks/LTX-2
- 固定 revision：`400fd31054597515f47125691032c04b1c3ee24e`
- 修改记录：[UPSTREAM.md](https://github.com/ZeyuLing/HFTrainer/blob/main/hftrainer/models/ltx_video/UPSTREAM.md)
- 完整许可证：[LTX-2.x Community License Agreement](https://github.com/ZeyuLing/HFTrainer/blob/main/hftrainer/models/ltx_video/LICENSE.ltx-2.x)
- 仓库第三方声明：[THIRD_PARTY_NOTICES.md](https://github.com/ZeyuLing/HFTrainer/blob/main/THIRD_PARTY_NOTICES.md)

LTX 许可证不是 Apache/MIT 式宽松许可证，其中包含用途限制、再分发义务、修改文件声明要求和商业许可条件。使用或分发前必须阅读完整协议。所有修改过的 Python 文件顶部都带有变更声明。

## 支持面与验证边界

| 能力 | 入口 | 本仓库验证 |
| --- | --- | --- |
| 蒸馏版文/图条件同步音视频生成 | `LTXVideoBundle` + `LTXVideoPipeline` | config、权重角色、shape、backend 构造与调用参数 |
| Dev 两阶段 + 官方/用户 LoRA | 同一接口，`mode='dev_two_stage'` | 两类 LoRA 的独立角色与顺序、guidance 参数、调用合约 |
| LoRA 训练 | `LTXVideoTrainer` | 本地 config 解析、LoRA 接入、checkpoint/resume 映射、managed runner 分发 |
| 数据预处理 | `hftrainer-ltx-preprocess` | 随包脚本定位、完整参数和权重角色校验 |
| 本地 Gemma 文本路径 | `hftrainer/models/ltx_video/network/text_encoders/gemma` | tiny 前反向、tokenizer/processor、hidden-state 合约、checkpoint 命名 |
| 真实 22B 生成/训练 | 同一套公开接口 | **没有在当前开发机器实跑** |

最后一行很重要：测试不能证明 22B 模型质量、吞吐、训练收敛或真实 CUDA 全量分配成功。

LTX pipeline 支持图片作为视频生成条件；图片条件的 *Gemma prompt enhancement* 是另一项可选功能。由于 Gemma vision tower 尚未完成本地化，该选项会明确报错，不会偷偷调用外部实现。正常文本编码和纯文本 prompt enhancement 使用本地 Gemma。

## 运行环境

先为机器选择正确的 CUDA PyTorch wheel，再安装本地源码所需的额外工具：

```bash
python -m pip install -e ".[ltx-video-inference]"
python -m pip install -e ".[ltx-video-train]"
python -m pip install -e ".[ltx-video]"  # 训练与推理

# 仅在启用对应功能时安装：
python -m pip install -e ".[ltx-video-integrations]"  # W&B / Hub 发布
python -m pip install -e ".[ltx-video-hdr]"           # EXR / HDR 媒体
```

extra 只包含媒体/科学计算支持库，不包含外部 LTX 或模型框架。构造 22B 图之前，本地 runtime 会检查所需 PyTorch compiler 能力并给出明确错误。当前对外支持的训练路径是 Linux + NVIDIA CUDA；真实显存取决于分辨率、帧数、音频、validation、精度和 offload 设置。

## 权重包

接受 gated 模型条款并下载后，将分体权重放在同一根目录：

```bash
export LTX25_CHECKPOINT_ROOT="$PWD/checkpoints/LTX-2.5"
```

示例 config 使用以下角色：

| 组件 | 作用 |
| --- | --- |
| distilled transformer | 固定 schedule 的快速生成 |
| dev transformer | LoRA 训练与引导式两阶段生成 |
| 带 LTX projection 的 Gemma 4 encoder | prompt 特征；不能直接替换为原生 Gemma |
| video VAE | 视频 latent 编解码 |
| audio VAE/vocoder | 同步音频 latent 编解码 |
| spatial upsampler | 第二阶段 latent 上采样 |
| duration head | 可选的 prompt 时长预测 |
| 官方 distilled LoRA | Dev 两阶段中必需的阶段衔接 |
| 用户 LoRA | 训练产生的适配权重 |

权重角色会在昂贵的模型加载前校验。Comfy 专用 INT8-convrot 权重会被本地 native 路径拒绝。

## 蒸馏版推理

```bash
hftrainer-ltx-infer \
  configs/ltx_video/infer_ltx_video_2_5_distilled.py \
  --prompt "A paper boat drifts through a rain-filled street at dusk." \
  --output outputs/ltx_video_2_5/distilled.mp4
```

可用 `--image frame.jpg` 添加条件帧；配置 duration head 后可用 `--auto-duration`。蒸馏 schedule 是固定的，negative prompt 和自定义步数会被直接拒绝，而不是静默忽略。

当前两阶段输出合约要求高宽能被 64 整除，并满足 `num_frames % 8 == 1`。

## 数据预处理

随包脚本接受 CSV、JSON 或 JSONL。最小 JSON item：

```json
{
  "caption": "A handheld shot follows a cyclist through a quiet alley.",
  "video": "videos/cyclist.mp4"
}
```

计算 latent 与文本特征：

```bash
hftrainer-ltx-preprocess data/ltx_video_2_5/dataset.jsonl \
  --resolution-buckets 960x544x49 \
  --model-path "$LTX25_CHECKPOINT_ROOT/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
  --text-encoder-path "$LTX25_CHECKPOINT_ROOT/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --video-vae-path "$LTX25_CHECKPOINT_ROOT/vae/ltx-2.5-video-vae-bf16.safetensors" \
  --audio-vae-path "$LTX25_CHECKPOINT_ROOT/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --output-dir data/ltx_video_2_5/.precomputed
```

不存在 `--ltx-repo` 参数。只有当训练策略也关闭生成音频时才使用 `--skip-audio`。切换权重角色、bucket 或 text encoder 后用 `--overwrite` 重新处理。

## LoRA 训练

```bash
export LTX25_PREPROCESSED_DATA="$PWD/data/ltx_video_2_5/.precomputed"
export HFTRAINER_WORK_DIR="$PWD/outputs/training/ltx_video_2_5_lora"

hftrainer-train configs/ltx_video/train_ltx_video_2_5_lora.py
```

`LTXVideoTrainer` 是 managed trainer，因为 LTX 的数据缓存、validation、checkpoint 生命周期高度耦合。“managed”不表示调用外部包：实现随包位于
`hftrainer.trainers.ltx_video.native`，使用 HFTrainer 本地 LoRA，并由常规 runner builder 选择。

示例只是起始 recipe，不是最优 benchmark 配置。

## 使用训练 LoRA 做 Dev 推理

```bash
export LTX25_USER_LORA="$PWD/outputs/training/ltx_video_2_5_lora/checkpoints/<saved-lora>.safetensors"

hftrainer-ltx-infer \
  configs/ltx_video/infer_ltx_video_2_5_dev_lora.py \
  --prompt "A slow dolly shot moves through an art studio while rain taps the windows." \
  --negative-prompt "blurry, distorted, low quality, artifacts" \
  --num-inference-steps 30 \
  --output outputs/ltx_video_2_5/dev_lora.mp4
```

该模式分别使用 dev transformer、必需的官方 distilled LoRA 和用户 LoRA，支持 guidance、negative prompt 和可配置步数。

## 常见问题

- **缺少支持库：**安装对应 LTX extra；异常应明确指出缺失的媒体/runtime 库。
- **出现禁用模型包 import：**这是源码问题，不能靠安装该模型包绕过，应报告具体 import 路径。
- **权重角色错误：**检查 dev/distilled transformer，以及带 projection 的 Gemma/LTX encoder 文件。
- **CUDA OOM：**降低 bucket、帧数和 validation，开启 gradient checkpointing，并选择合适的精度/offload。
- **合约测试通过但真实生成失败：**提供 GPU、driver、CUDA/PyTorch、HFTrainer commit、完整命令、权重文件名和第一段堆栈。合约测试不会分配完整 22B 模型。
