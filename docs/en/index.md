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

## Public models and data

All eight model families have processed Hugging Face checkpoints, public-data training recipes and a common English model-card format. Start with the [model catalog](models/index.md), [public datasets](models/demo-data.md) and [release standard](models/standard.md).

Full-weight H3 T2VA and LTX dev generation were compared with independent official implementations on an H200. The cards distinguish these base-model comparisons from short public-data training, reduced-resolution checks and untested settings. Short runs do not establish convergence.
