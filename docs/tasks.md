# Supported Tasks

| Task | ModelBundle | Trainer | Pipeline | Example Models |
|---|---|---|---|---|
| Classification | `ViTBundle` | `ClassificationTrainer` | `ClassificationPipeline` | ViT, DeiT, Swin |
| Text-to-Image | `SD15Bundle` | `SD15Trainer` | `SD15Pipeline` | SD1.5, SDXL |
| LLM SFT | `CausalLMBundle` | `CausalLMTrainer` | `CausalLMPipeline` | TinyLlama, LLaMA, Qwen |
| Text-to-Video | `WanBundle` | `WanTrainer` | `WanPipeline` | WAN 1.3B/14B |

## Demo Configs

Each task has a demo config for smoke testing:

| Task | Config | Requirements |
|---|---|---|
| Classification | `configs/classification/vit_base_demo.py` | CPU or GPU |
| Text-to-Image | `configs/text2image/sd15_demo.py` | GPU with fp16 |
| LLM SFT | `configs/llm/llama_sft_demo.py` | GPU with bf16 |
| Text-to-Video | `configs/text2video/wan_demo.py` | GPU with bf16 |

## Validation Output Convention

Each task's `val_step` returns a plain Python dict with task-specific keys:

| Task | Required Keys |
|------|--------------|
| text2image | `preds` (Tensor), `gts` (Tensor, optional), `prompts` (list[str]) |
| text2video | `preds` (Tensor[B,T,C,H,W]), `prompts` (list[str]) |
| llm | `preds` (list[str]), `gts` (list[str]), `input_prompts` (list[str]) |
| classification | `preds` (Tensor[B]), `scores` (Tensor[B,C]), `gts` (Tensor[B]) |
| detection | `pred_boxes/scores/labels` (list[Tensor]), `gt_boxes/labels` (list[Tensor]), `metas` |

Evaluators and Visualizers consume these dicts directly -- no special container classes needed.
