# Design Overview

HF-Trainer is built around a small set of ideas:

- config-driven construction with MMEngine `Config` and `Registry`
- `accelerate` as the runtime layer
- direct use of HuggingFace-native components
- shared task logic through `ModelBundle`

## Design Pages

- [ModelBundle](model_bundle.md)
- [Accelerate Integration & Per-Module Isolation](accelerate_integration.md)
- [Checkpointing](checkpoint.md)
- [Hooks](hooks.md)
- [LoRA](../lora.md)
- [Multi-Optimizer](multi_optimizer.md)
- [Datasets](dataset.md)
- [Evaluation and Visualization](evaluation.md)
