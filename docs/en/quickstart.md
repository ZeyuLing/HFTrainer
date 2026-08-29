# Quick Start

## Verified Smoke Test

The most reliable first run is the classification demo:

```bash
python3 tools/train.py configs/vit/vit_base_demo.py
```

This exercises:

- config loading
- dataloader construction
- model forward/backward
- checkpoint saving
- validation

## Other Demo Configs

```bash
python3 tools/train.py configs/sd15/sd15_demo.py
python3 tools/train.py configs/llama/llama_sft_demo.py
python3 tools/train.py configs/llama/llama_lora_demo.py
python3 tools/train.py configs/wan/wan_demo.py
```

These require the corresponding checkpoints in `checkpoints/` and enough GPU memory for the task.

LoRA quick start:

```bash
python3 tools/train.py configs/llama/llama_lora_demo.py
python3 tools/infer.py \
  --config configs/llama/llama_lora_demo.py \
  --checkpoint work_dirs/llama_lora_smoke/checkpoint-iter_10 \
  --merge-lora \
  --prompt "Name one primary color."
```

## Inference

Classification:

```bash
python3 tools/infer.py \
  --config configs/vit/vit_base_demo.py \
  --checkpoint work_dirs/vit_smoke/checkpoint-iter_10 \
  --input data/classification/demo/images/cat/cat_000.jpg \
  --device cpu
```

Text-to-image:

```bash
python3 tools/infer.py \
  --config configs/sd15/sd15_demo.py \
  --checkpoint work_dirs/sd15_smoke/checkpoint-iter_10 \
  --prompt "a red cat on a mat"
```
