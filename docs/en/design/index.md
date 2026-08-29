# Design Overview

HFTrainer is built around a small set of ideas:

- config-driven construction with MMEngine `Config` and `Registry`
- `accelerate` as the runtime layer
- repository-owned model components resolved through local registries
- shared training/inference operations through `ModelBundle`
- implementation-owned artifact schemas and local LoRA
- external model-package imports rejected by structural tests

## Design Pages

- [ModelBundle](model_bundle.md)
- [Checkpointing](checkpoint.md)
- [Hooks](hooks.md)
- [LoRA](../lora.md)
- [Multi-Optimizer](multi_optimizer.md)
- [Datasets](dataset.md)
- [Evaluation and Visualization](evaluation.md)
