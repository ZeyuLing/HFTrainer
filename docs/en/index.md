# HF-Trainer Docs

HF-Trainer is a config-driven framework for training HuggingFace-native models with `accelerate`. The current codebase focuses on runnable task stacks plus a small set of framework primitives: `ModelBundle`, `Trainer`, `Pipeline`, `AccelerateRunner`, hooks, evaluators, and visualizers.

## Start Here

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [Integration Guide](integration.md)
- [API Reference](api_reference.md)
- [Memory and Precision](memory.md)
- [LoRA](lora.md)
- [Architecture](architecture.md)
- [Distributed Training](distributed.md)
- [Experiment Directory](experiment_dir.md)
- [Task Matrix](tasks.md)
- [Comparison](comparison.md)

## Design Notes

- [Design Overview](design/index.md)
- [ModelBundle](design/model_bundle.md)
- [Checkpointing](design/checkpoint.md)
- [Hooks](design/hooks.md)
- [Multi-Optimizer](design/multi_optimizer.md)
- [Datasets](design/dataset.md)
- [Evaluation and Visualization](design/evaluation.md)

## Project Status

Runnable demos:

- Classification
- Text-to-image
- Causal LM SFT
- Causal LM LoRA
- Text-to-video
- Motion generation (PRISM, VerMo)
- StyleGAN2-style GAN training
- DMD-style distillation
