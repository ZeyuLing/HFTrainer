# MiniMax-H3

HFTrainer 现在维护 MiniMax-H3 Base 768p 的仓库内原生实现，覆盖文本、首尾帧和有序多模态参考输入到同步视频/双声道音频的生成。运行时不会 import Diffusers、Transformers、Tokenizers、PEFT，也不依赖第二份 MiniMax 源码 checkout。

```text
hftrainer/models/minimax_h3/network/   Transformer、Qwen3-VL、tokenizer、
                                       processor、视频/音频 VAE、scheduler
hftrainer/models/minimax_h3/bundle.py  组件、artifact 边界与原子操作
hftrainer/pipelines/minimax_h3/        T2VA、FL2VA、Ref2VA 推理图
hftrainer/trainers/minimax_h3/         实验性缓存特征 RF 训练目标
hftrainer/datasets/synchronized_audio_video/
                                       H3 缓存特征数据合约
configs/minimax_h3/                    推理与 LoRA 训练配置
```

## 支持范围

| 能力 | 本地状态 |
| --- | --- |
| T2VA | 使用公开 `transformer/` 权重分区 |
| 首帧、尾帧、首尾帧到同步音视频 | 使用公开 `transformer/` 权重分区 |
| 有序图片/视频/音频参考 | 使用独立 `transformer_ref/` 权重分区 |
| Qwen3-VL-32B 条件器 | 本地 vision/text 前向，读取 hidden state 50 |
| 视频/音频 codec | 本地 24 通道视频 VAE；32 通道、32 kHz 音频 VAE |
| 视频/音频 flow schedule | 本地 shift-12 与 shift-3 scheduler |
| 微调 | 实验性缓存特征 full/LoRA Transformer 训练 |
| H3-Context-IR、Regenerate-2K、托管 2K 阶段 | 未公开本地权重，明确不支持 |
| sparse attention kernel | 上游未公开，本地使用 full attention |

H3 是联合音视频 rectified-flow 模型，不是自回归语言模型；没有 CFG，也不接受 negative prompt。Transformer 会把文本/视觉条件、参考音视频和目标音视频行打包成一个非因果序列。

## 来源与许可证

- 官方仓库：https://github.com/MiniMax-AI/MiniMax-H3
- 官方权重：https://huggingface.co/MiniMaxAI/MiniMax-H3
- 固定来源与修改记录：`hftrainer/models/minimax_h3/UPSTREAM.md`
- Apache-2.0 参考代码许可证：`hftrainer/models/minimax_h3/LICENSE.apache-2.0`
- 完整模型协议：`hftrainer/models/minimax_h3/LICENSE.minimax-h3`
- 模型 NOTICE：`hftrainer/models/minimax_h3/NOTICE.minimax-h3`

MiniMax 模型材料受 MiniMax H3 Community License Agreement 约束，并不是宽松开源许可证。协议包含适用地域排除、用途限制、再分发义务和额外商业条款。下载或使用权重前必须自行阅读并接受完整协议。HFTrainer 不随源码仓库分发权重、tokenizer 资源或上游 config。

本地代码参考了固定 commit 的 Apache-2.0 Diffusers 与 Transformers 实现，并用 HFTrainer 自有的配置、加载、预处理和编排替换外部框架机制；具体 commit 见修改记录。

## 安装与下载

先选择适合机器的 CUDA PyTorch wheel，然后安装媒体/下载工具并固定权重 revision：

```bash
python -m pip install -e ".[minimax-h3]"

hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include "model_index.json" "modular_model_index.json" \
    "processor/*" "tokenizer/*" "text_encoder/*" \
    "vae/*" "audio_vae/*" "scheduler/*" "audio_scheduler/*" \
    "transformer/*" "transformer_ref/*" \
  --local-dir checkpoints/MiniMax-H3

export MINIMAX_H3_ROOT="$PWD/checkpoints/MiniMax-H3"
```

上述 include 列表只下载 HFTrainer 使用的共享 Diffusers 格式组件和两个本地
Transformer 分区，并明确排除同一仓库中额外提供的原始格式 `FL2VA/`、
`Ref2VA/` 目录，避免重复下载一整份大体积 checkpoint。

权重体积很大：两个互斥的 Transformer 分区各约 66 GB，Qwen3-VL 条件器约 67 GB，两个 codec 合计约 11 GB。加载前需要规划磁盘、主存和加速器显存。本地 loader 会拒绝 shard 缺失、重复 key、shape 不符和低覆盖率部分加载。

完整 Transformer、条件器和 codec 通常无法放进同一张 GPU，推理配置支持按组件显式放置：

```bash
export MINIMAX_H3_TRANSFORMER_DEVICE=cuda:0
export MINIMAX_H3_CONDITIONER_DEVICE=cuda:1
export MINIMAX_H3_CODEC_DEVICE=cuda:2
```

`MINIMAX_H3_LOAD_DEVICE` 是三者的公共 fallback。拆分放置时不要再传通用
CLI 的 `--device`，该参数会把整个 bundle 搬到同一设备。本地公共 loader
会明确拒绝 `device_map`，不会假装完成逐层切分；当前接入边界是上述组件级
放置，不包含 layer-wise dispatch。

## 推理

纯文本使用 FL2VA 权重分区但不提供关键帧：

```bash
hftrainer-infer \
  --config configs/minimax_h3/infer_h3_base_fl2va.py \
  --prompt "A paper boat drifts down a narrow stream; water and birds are audible." \
  --mode t2va --duration 5 \
  --output outputs/minimax_h3/t2va.mp4
```

首尾帧条件使用同一个权重分区：

```bash
hftrainer-infer \
  --config configs/minimax_h3/infer_h3_base_fl2va.py \
  --prompt "The camera slowly circles the subject while wind moves the leaves." \
  --mode fl2va \
  --first-frame assets/start.png --last-frame assets/end.png \
  --output outputs/minimax_h3/fl2va.mp4
```

多模态参考必须换成独立 Ref2VA 权重；参考顺序具有语义，CLI 会保持传入顺序：

```bash
hftrainer-infer \
  --config configs/minimax_h3/infer_h3_base_ref2va.py \
  --prompt "Use Picture 1 for the character and Video 1 for motion and camera." \
  --mode ref2va \
  --reference-image assets/character.png \
  --reference-video assets/motion.mp4 \
  --duration 5 \
  --output outputs/minimax_h3/ref2va.mp4
```

输出固定为 24 FPS 视频和 32 kHz 双声道音频。当前可执行公开路径限制在 5–15 秒；帧数会向上对齐到视频 VAE 合法的 `17*n+5`。高宽必须同时是 32 的倍数；省略时使用公开模型的 768 短边 canvas 规则。Ref2VA 最多接收 9 张图、3 个视频、3 段音频、合计 12 个参考，不能只提供音频参考。Ref2VA 的 Python 调用必须显式传入 `duration` 或 `num_frames`，避免沿用 T2VA/FL2VA 的 124 帧默认值而静默截断参考音轨。

Python pipeline 支持 `output_type="pt"`、`"np"`、`"pil"` 和 `"latent"`。
Tensor 视频布局为 `B,T,C,H,W`，NumPy 为 `B,T,H,W,C`，PIL 为 batch/frame
两层列表。与固定上游 API 一致，所有非 latent 视频输出下的解码音频都保持
为 float32 CPU tensor `B,2,S`。音频保留 codec 按 800-sample hop 产生的
完整长度，不会按非整数视频帧时长再次截断。HFTrainer 有意将
`output_type` 默认为 `"pt"`，便于 CLI 直接封装 tensor；固定的 Diffusers
modular pipeline 默认为 `"pil"`。

需要确定性重放时，`latents` 可传入 `[1,24,T,H,W]` 的预生成视频噪声，
`audio_latents` 可传入 `[2,32,A]` 的双声道 channel-major 音频噪声。
传入任一路只会跳过该路的随机抽样，其余抽样仍保持公开实现的“条件 → 视频 → 音频”顺序。
`attention_kwargs` 会在每次 Transformer 前向中原样透传。

## 实验性训练

MiniMax 发布了权重，但没有公开完整训练 recipe。因此 `MiniMaxH3Trainer` 实现的是由公开 scheduler 推导出的 data-ward rectified-flow 目标：

```text
x_t = t*x_0 + (1-t)*noise
target = x_0 - noise
```

这是明确标注的实验性目标，不声明官方 recipe 或收敛一致。推荐先缓存冻结特征，使 33B Transformer 训练时不需要同时常驻 32B 条件器和两个 VAE。

JSONL manifest 每行指向一个 `.safetensors` 或 `.pt`：

```json
{"feature_file":"000001.safetensors","keyframe_anchors":[],"reference_geometries":[]}
```

缓存文件包含：

| Tensor | Shape | 含义 |
| --- | --- | --- |
| `video_latents` | `[24,T,H,W]` | 归一化 clean 视频 latent |
| `audio_latents` | `[2,32,L]` | 归一化 clean 双声道音频 latent |
| `prompt_embeds` | `[N,5120]` | 本地 Qwen3-VL hidden state 50 |
| `text_token_tags` | `[N]` | 文本/视觉模态 tag |
| `condition_video_rows` | 可选 `[Nv,Dv]` | FL2VA/Ref2VA clean 视频条件行 |
| `condition_audio_rows` | 可选 `[Na,32]` | Ref2VA clean 音频条件行 |

运行 LoRA 配置：

```bash
export MINIMAX_H3_CACHE_MANIFEST="$PWD/data/minimax_h3/train.jsonl"
export HFTRAINER_WORK_DIR="$PWD/outputs/training/minimax_h3_lora"
hftrainer-train configs/minimax_h3/train_h3_base_lora.py
```

训练 checkpoint 只保存本地 adapter 张量及 HFTrainer 元数据。用于生成时，
应通过匹配的 adapter-aware 推理配置加载，并在加载后合并：

```bash
hftrainer-infer \
  --config configs/minimax_h3/infer_h3_base_fl2va_lora.py \
  --checkpoint outputs/training/minimax_h3_lora/checkpoint-iter_2000 \
  --merge-lora \
  --mode t2va \
  --prompt "一名鼓手在舞台上演奏" \
  --duration 5 \
  --output outputs/minimax_h3/lora.mp4
```

`MiniMaxH3Bundle.save_pretrained(...)` 会先把当前本地 LoRA 合并进
Transformer，再导出可独立回载的完整 artifact。存在活动 adapter 时传入
`merge_lora=False` 会直接报错，不会写出无法回载的坏产物。需要恢复 LoRA
训练时，应保留原 HFTrainer checkpoint。

该产物有意采用 HFTrainer bundle 格式，并不是可直接替代 Diffusers 顶层
artifact 的兼容层。组件级 `from_pretrained(...)` 只接受本地目录以及显式的
`device`/dtype 放置；Hub 解析、`device_map`、checkpoint `variant` 和
`use_safetensors` 选择都会明确报错，而不会被静默近似。组件加载默认使用
meta-device 低内存路径。HFTrainer 还为本地 audio VAE 增加了 gradient
checkpointing 训练扩展，但不会改变公开 audio-VAE checkpoint 的 key schema。

同一 minibatch 的 packed row geometry 必须完全一致。应按分辨率、latent 时长、prompt presentation 长度和参考布局分桶，或直接使用 `batch_size=1`。

## 验证边界

仓库测试覆盖 tiny Transformer、scheduler、VAE、Qwen/tokenizer、layout、loss/backward、pipeline、缓存与 artifact 合约，并会在主动阻断外部模型包的进程中导入。开发阶段的数值对照使用固定上游参考，但上游包不是运行依赖。

这些检查**不能**证明完整权重的生成质量、吞吐、收敛，也不能证明当前开发机成功分配了 33B+32B 全量模型。正式发布验收仍应在目标多 GPU 机器上使用固定完整权重跑通三种公开模式，并人工检查最终封装音视频。
