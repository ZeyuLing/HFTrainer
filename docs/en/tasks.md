# Task Matrix

## Runnable Today

| Task | Bundle | Trainer | Pipeline | Example Config |
| --- | --- | --- | --- | --- |
| Classification | `ViTBundle` | `ClassificationTrainer` | `ClassificationPipeline` | `configs/classification/vit_base_demo.py` |
| Text-to-image | `SD15Bundle` | `SD15Trainer` | `SD15Pipeline` | `configs/text2image/sd15_demo.py` |
| Causal LM SFT | `CausalLMBundle` | `CausalLMTrainer` | `CausalLMPipeline` | `configs/llm/llama_sft_demo.py` |
| Causal LM LoRA | `CausalLMBundle` | `CausalLMTrainer` | `CausalLMPipeline` | `configs/llm/llama_lora_demo.py` |
| Text-to-video | `WanBundle` | `WanTrainer` | `WanPipeline` | `configs/text2video/wan_demo.py` |
| Motion generation (PRISM) | `PrismBundle` | `PrismTrainer` | `PrismPipeline` | `configs/prism/prism_1b_tp2m_motionhub.py` |
| Motion generation / understanding (VerMo) | `VermoBundle` | `VermoTrainer` | `VermoPipeline` | `configs/vermo/vermo_pretrain_4k_llama1b_wavtokenizer.py` |
| GAN | `StyleGAN2Bundle` | `GANTrainer` | `StyleGAN2Pipeline` | `configs/gan/gan_demo.py` |
| DMD one-step distillation | `DMDBundle` | `DMDTrainer` | `DMDPipeline` | `configs/distillation/dmd_demo.py` |

## Validation Output Convention

- Classification: `preds`, `scores`, `gts`, optional `metas`
- Text-to-image: `preds`, `prompts`, optional `gts`
- Text-to-video: `preds`, `prompts`
- Motion generation: task-specific motion artifacts or raw multimodal responses, depending on the pipeline
- LLM: `preds`, `gts`, `input_prompts`, optional `loss_lm`
