# HF-Trainer

A unified, config-driven training framework built on the HuggingFace ecosystem. Combines MMEngine's declarative config system with Accelerate-native distributed training, and uses a `ModelBundle` abstraction to share forward logic between training and inference.

## Features

- **Config-Driven, Registry-Based** -- Define models, datasets, optimizers, hooks, and more through `.py` config files with `_base_` inheritance (MMEngine style). No boilerplate, just declare what you need.

- **Accelerate-Native** -- All distributed training (DDP, FSDP, DeepSpeed), mixed precision, and gradient accumulation are managed by HuggingFace Accelerate. No custom distributed wrappers.

- **ModelBundle = Shared Core** -- Trainer and Pipeline share the same `ModelBundle` instance. All forward functions (`encode_text`, `predict_noise`, `decode_latent`, etc.) are written once and reused in both training and inference, eliminating code duplication.

- **Per-Module Control** -- Each sub-module (text_encoder, vae, transformer, etc.) independently controls `trainable`, `save_ckpt`, and LoRA injection via config. Frozen modules are skipped during checkpoint I/O.

- **HuggingFace-First** -- Directly uses `diffusers`, `transformers`, and `peft` classes with no extra wrappers.

- **Unified Checkpoint** -- Single `load_from` + `load_scope` API for both resume and transfer learning. `auto_resume=True` auto-detects the latest checkpoint for seamless cluster preemption recovery.

- **Multi-Task Support** -- Classification, text-to-image, text-to-video, LLM SFT, and detection out of the box, with demo configs and smoke tests for each.

- **Multi-Optimizer** -- Named optimizer dict for GAN training and other multi-optimizer scenarios.

## Quick Example

```bash
# ViT classification smoke test (no GPU required)
python tools/train.py configs/classification/vit_base_demo.py

# Distributed training with 8 GPUs
bash tools/dist_train.sh configs/text2video/wan_demo.py 8
```

## Getting Started

- [Installation](installation.md) -- Install the package and download pretrained checkpoints
- [Quick Start](quickstart.md) -- Run smoke tests and inference examples
- [Distributed Training](distributed.md) -- DDP, FSDP, DeepSpeed, single GPU
- [Architecture](architecture.md) -- How the framework is structured

## Design Documentation

For in-depth design rationale (bilingual Chinese/English):

- [Design Overview](design/index.md) -- Motivation and design principles
- [ModelBundle](design/model_bundle.md) -- Per-module control and Trainer/Pipeline sharing
- [Checkpoint](design/checkpoint.md) -- Unified `load_from` / `load_scope`
- [Multi-Optimizer](design/multi_optimizer.md) -- GAN support
- [Dataset](design/dataset.md) -- Dataset directory structure
- [Evaluation](design/evaluation.md) -- Evaluator/Visualizer dict interface
