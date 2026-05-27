# HF-Trainer

A unified, config-driven training framework built on the HuggingFace ecosystem. Combines MMEngine's declarative config system with Accelerate-native distributed training, and uses a `ModelBundle` abstraction to share forward logic between training and inference.

## Features

- **Config-Driven, Registry-Based** -- `.py` config files with `_base_` inheritance (MMEngine style)
- **Accelerate-Native** -- DDP, FSDP, DeepSpeed, mixed precision, gradient accumulation via HuggingFace Accelerate
- **ModelBundle = Shared Core** -- Trainer and Pipeline share the same forward functions, written once
- **Per-Module Control** -- Each sub-module independently controls `trainable`, `save_ckpt`, and LoRA via config
- **HuggingFace-First** -- Directly uses `diffusers`, `transformers`, and `peft` with no extra wrappers
- **Unified Checkpoint** -- Single `load_from` + `load_scope` API; `auto_resume=True` for cluster preemption recovery

## Installation

```bash
git clone <repo-url> && cd hf_trainer
pip install -e .
```

## Quick Start

```bash
# ViT classification smoke test (no GPU required)
python tools/train.py configs/classification/vit_base_demo.py

# Distributed training with 8 GPUs
bash tools/dist_train.sh configs/text2video/wan_demo.py 8
```

## Supported Tasks

| Task | ModelBundle | Trainer | Pipeline | Example Models |
|---|---|---|---|---|
| Classification | `ViTBundle` | `ClassificationTrainer` | `ClassificationPipeline` | ViT, DeiT, Swin |
| Text-to-Image | `SD15Bundle` | `SD15Trainer` | `SD15Pipeline` | SD1.5, SDXL |
| LLM SFT | `CausalLMBundle` | `CausalLMTrainer` | `CausalLMPipeline` | TinyLlama, LLaMA, Qwen |
| Text-to-Video | `WanBundle` | `WanTrainer` | `WanPipeline` | WAN 1.3B/14B |

## Documentation

For full documentation, see the [docs/](docs/) directory:

- [Installation](docs/installation.md) -- Install the package and download pretrained checkpoints
- [Quick Start](docs/quickstart.md) -- Smoke tests and inference examples
- [Distributed Training](docs/distributed.md) -- DDP, FSDP, DeepSpeed, single GPU
- [Experiment Directory](docs/experiment_dir.md) -- Work directory layout, checkpoint management, auto-resume
- [Architecture](docs/architecture.md) -- How the framework is structured
- [Supported Tasks](docs/tasks.md) -- Task table and val output conventions
- [Comparison](docs/comparison.md) -- Comparison with MMEngine and HuggingFace Trainer
- [Design Docs](docs/design/index.md) -- In-depth design rationale (bilingual Chinese/English)

To serve the docs locally with MkDocs:

```bash
pip install mkdocs-material
mkdocs serve
```

## License

Apache 2.0
