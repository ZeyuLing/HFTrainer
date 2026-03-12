# 任务矩阵

## 当前可运行

| 任务 | Bundle | Trainer | Pipeline | 示例 Config |
| --- | --- | --- | --- | --- |
| 图像分类 | `ViTBundle` | `ClassificationTrainer` | `ClassificationPipeline` | `configs/classification/vit_base_demo.py` |
| 文生图 | `SD15Bundle` | `SD15Trainer` | `SD15Pipeline` | `configs/text2image/sd15_demo.py` |
| Causal LM SFT | `CausalLMBundle` | `CausalLMTrainer` | `CausalLMPipeline` | `configs/llm/llama_sft_demo.py` |
| Causal LM LoRA | `CausalLMBundle` | `CausalLMTrainer` | `CausalLMPipeline` | `configs/llm/llama_lora_demo.py` |
| 文生视频 | `WanBundle` | `WanTrainer` | `WanPipeline` | `configs/text2video/wan_demo.py` |
| 动作生成（PRISM） | `PrismBundle` | `PrismTrainer` | `PrismPipeline` | `configs/prism/prism_1b_tp2m_motionhub.py` |
| 动作生成 / 理解（VerMo） | `VermoBundle` | `VermoTrainer` | `VermoPipeline` | `configs/vermo/vermo_pretrain_4k_llama1b_wavtokenizer.py` |
| GAN | `StyleGAN2Bundle` | `GANTrainer` | `StyleGAN2Pipeline` | `configs/gan/gan_demo.py` |
| DMD 一步蒸馏 | `DMDBundle` | `DMDTrainer` | `DMDPipeline` | `configs/distillation/dmd_demo.py` |

## Validation 输出约定

- 分类：`preds`、`scores`、`gts`、可选 `metas`
- 文生图：`preds`、`prompts`、可选 `gts`
- 文生视频：`preds`、`prompts`
- 动作生成：按任务返回动作产物或原始多模态响应文本
- LLM：`preds`、`gts`、`input_prompts`、可选 `loss_lm`
