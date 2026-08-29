# 任务矩阵

## 当前可运行

| 任务 | Bundle | Trainer | Pipeline | 示例 Config | 验证状态 |
| --- | --- | --- | --- | --- | --- |
| 图像分类 | `ViTBundle` | `ClassificationTrainer` | `ClassificationPipeline` | `configs/vit/vit_base_demo.py` | 内置 smoke 路径 |
| 文生图 | `SD15Bundle` | `SD15Trainer` | `SD15Pipeline` | `configs/sd15/sd15_demo.py` | 内置 smoke 路径 |
| Causal LM SFT | `LlamaBundle` | `CausalLMTrainer` | `CausalLMPipeline` | `configs/llama/llama_sft_demo.py` | 内置 smoke 路径 |
| Causal LM LoRA | `LlamaBundle` | `CausalLMTrainer` | `CausalLMPipeline` | `configs/llama/llama_lora_demo.py` | 内置 smoke 路径 |
| 文生视频 | `WanBundle` | `WanTrainer` | `WanPipeline` | `configs/wan/wan_demo.py` | 内置 smoke 路径 |
| LTX-2.5 蒸馏同步音视频推理 | `LTXVideoBundle` | 不适用 | `LTXVideoPipeline` | `configs/ltx_video/infer_ltx_video_2_5_distilled.py` | 本地 registry/config/checkpoint 合约；未实跑 22B GPU |
| LTX-2.5 LoRA 训练 + Dev 推理 | `LTXVideoBundle` | `LTXVideoTrainer`（managed 本地循环） | `LTXVideoPipeline` | `configs/ltx_video/train_ltx_video_2_5_lora.py` | 随包 trainer/preprocess 合约与 tiny Gemma 路径；未实跑 22B GPU |
| GAN | `StyleGAN2Bundle` | `StyleGAN2Trainer` | `StyleGAN2Pipeline` | `configs/stylegan2/stylegan2_demo.py` | 框架参考实现 |
| DMD 一步蒸馏 | `DMDBundle` | `DMDTrainer` | `DMDPipeline` | `configs/dmd/dmd_demo.py` | 框架参考实现 |

“内置 smoke 路径”表示仓库为该任务提供降规模启动检查；“框架参考实现”表示用于
展示集成结构，不声明 benchmark 复现。LTX 的 model、trainer 与 pipeline 源码均
随仓库提供；轻量测试不会分配 gated 22B 权重，只验证本地
API/config/checkpoint 边界。详见
[LTX-Video 2.5 指南](models/ltx_video_2_5.md)。

## Validation 输出约定

- 分类：`preds`、`scores`、`gts`、可选 `metas`
- 文生图：`preds`、`prompts`、可选 `gts`
- 文生视频：`preds`、`prompts`
- LLM：`preds`、`gts`、`input_prompts`、可选 `loss_lm`

LTX pipeline 返回 video/audio 对象或已编码文件路径，并包含帧数、帧率、高宽、seed、
mode 与实际 tiling config。编码会消费 video iterator，因此保存文件后的结果中
`video=None`，不会返回一个已经被消费的 iterator。
