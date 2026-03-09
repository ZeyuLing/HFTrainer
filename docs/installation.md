# Installation

## Install the Package

```bash
git clone <repo-url> && cd hf_trainer
pip install -e .
```

Optional: install tensorboard for training visualization:

```bash
pip install tensorboard
```

## Checkpoint Downloads

The demo configs expect pretrained weights in `checkpoints/`. Download them all at once:

```bash
bash tools/download_checkpoints.sh
```

Or download individual models:

| Model | HuggingFace Hub ID | Command |
|---|---|---|
| ViT-Base | `google/vit-base-patch16-224` | `huggingface-cli download google/vit-base-patch16-224 --local-dir checkpoints/vit-base-patch16-224` |
| SD1.5 | `stable-diffusion-v1-5/stable-diffusion-v1-5` | `huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 --local-dir checkpoints/stable-diffusion-v1-5` |
| TinyLlama | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir checkpoints/TinyLlama-1.1B-Chat-v1.0` |
| WAN-1.3B | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir checkpoints/Wan2.1-T2V-1.3B-Diffusers` |

See [checkpoints/README.md](../checkpoints/README.md) for details.

## Key Dependencies

- `torch >= 2.0`
- `accelerate` -- distributed training, mixed precision, gradient accumulation
- `transformers` -- LLM, ViT, DETR, tokenizers, processors
- `diffusers` -- Stable Diffusion, WAN, noise schedulers
- `peft` -- LoRA, QLoRA
- `mmengine` -- Config (`.py` parsing, `_base_` inheritance), Registry (only these two components are used)
- `safetensors` -- fast checkpoint I/O
- `wandb` / `tensorboard` -- logging
