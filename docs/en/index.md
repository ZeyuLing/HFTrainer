# HFTrainer Docs

HFTrainer is a config-driven training and inference framework whose model
implementations live in this repository. `ModelBundle`, trainer, and pipeline
layers share one local component graph; Accelerate provides distributed
runtime orchestration rather than model definitions.

## Start Here

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [LTX-Video 2.5](models/ltx_video_2_5.md)
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
- StyleGAN2-style GAN training
- DMD-style distillation
- LTX-Video 2.5 distilled/dev inference and local managed LoRA training

The packaged LTX implementation is a modified snapshot pinned to one source
revision. Its local config/API contracts and tiny Gemma path are tested; the
repository test environment does not execute the gated 22B workflow.
