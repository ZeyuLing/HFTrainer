# Quick Start

## Verified Smoke Test

The most reliable first run is the classification demo:

```bash
python3 tools/train.py configs/classification/vit_base_demo.py
```

This exercises:

- config loading
- dataloader construction
- model forward/backward
- checkpoint saving
- validation

## Other Demo Configs

```bash
python3 tools/train.py configs/text2image/sd15_demo.py
python3 tools/train.py configs/llm/llama_sft_demo.py
python3 tools/train.py configs/llm/llama_lora_demo.py
python3 tools/train.py configs/text2video/wan_demo.py
```

These require the corresponding checkpoints in `checkpoints/` and enough GPU memory for the task.

LoRA quick start:

```bash
python3 tools/train.py configs/llm/llama_lora_demo.py
python3 tools/infer.py \
  --config configs/llm/llama_lora_demo.py \
  --checkpoint work_dirs/llama_lora_smoke/checkpoint-iter_10 \
  --merge-lora \
  --prompt "Name one primary color."
```

## Inference

Classification:

```bash
python3 tools/infer.py \
  --config configs/classification/vit_base_demo.py \
  --checkpoint work_dirs/vit_smoke/checkpoint-iter_10 \
  --input data/classification/demo/images/cat/cat_000.jpg \
  --device cpu
```

Text-to-image:

```bash
python3 tools/infer.py \
  --config configs/text2image/sd15_demo.py \
  --checkpoint work_dirs/sd15_smoke/checkpoint-iter_10 \
  --prompt "a red cat on a mat"
```
