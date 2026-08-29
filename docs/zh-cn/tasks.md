# 任务矩阵

## 当前可运行

| 任务 | Bundle | Trainer | Pipeline | 示例 Config | 验证状态 |
| --- | --- | --- | --- | --- | --- |
| 图像分类 | `ViTBundle` | `ClassificationTrainer` | `ClassificationPipeline` | `configs/classification/vit_base_demo.py` | 内置 smoke 路径 |
| 文生图 | `SD15Bundle` | `SD15Trainer` | `SD15Pipeline` | `configs/text2image/sd15_demo.py` | 内置 smoke 路径 |
| Causal LM SFT | `CausalLMBundle` | `CausalLMTrainer` | `CausalLMPipeline` | `configs/llm/llama_sft_demo.py` | 内置 smoke 路径 |
| Causal LM LoRA | `CausalLMBundle` | `CausalLMTrainer` | `CausalLMPipeline` | `configs/llm/llama_lora_demo.py` | 内置 smoke 路径 |
| 文生视频 | `WanBundle` | `WanTrainer` | `WanPipeline` | `configs/text2video/wan_demo.py` | 内置 smoke 路径 |
| LTX-2.5 蒸馏同步音视频推理 | `LTXVideoBundle` | 不适用 | `LTXVideoPipeline` | `configs/ltx_video/infer_ltx_video_2_5_distilled.py` | registry/config + mock 官方 API 合约；未实跑 22B GPU |
| LTX-2.5 LoRA 训练 + Dev 推理 | `LTXVideoBundle` | `LTXVideoTrainer`（managed） | `LTXVideoPipeline` | `configs/ltx_video/train_ltx_video_2_5_lora.py` | managed trainer/config + mock 官方 API 合约；未实跑 22B GPU |
| GAN | `StyleGAN2Bundle` | `GANTrainer` | `StyleGAN2Pipeline` | `configs/gan/gan_demo.py` | 框架参考实现 |
| DMD 一步蒸馏 | `DMDBundle` | `DMDTrainer` | `DMDPipeline` | `configs/distillation/dmd_demo.py` | 框架参考实现 |

“内置 smoke 路径”表示仓库为该任务提供降规模启动检查；“框架参考实现”表示用于
展示集成结构，不声明 benchmark 复现。LTX 的轻量测试不会分配 gated 22B 权重，
只验证固定版本的 API/config 合约。详见
[LTX-Video 2.5 指南](models/ltx_video_2_5.md)。

## Validation 输出约定

- 分类：`preds`、`scores`、`gts`、可选 `metas`
- 文生图：`preds`、`prompts`、可选 `gts`
- 文生视频：`preds`、`prompts`
- LLM：`preds`、`gts`、`input_prompts`、可选 `loss_lm`

LTX pipeline 返回 video/audio 对象或已编码文件路径，并包含帧数、帧率、高宽、seed、
mode 与实际 tiling config。编码会消费 video iterator，因此保存文件后的结果中
`video=None`，不会返回一个已经被消费的 iterator。
