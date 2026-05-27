# Comparison with Existing Frameworks

# 与现有框架的对比

## Overview / 概述

HF-Trainer takes the best of MMEngine's config system and combines it with HuggingFace Accelerate's distributed training, adding novel features like ModelBundle sharing and per-module checkpoint control.

HF-Trainer 取 MMEngine config 系统之长，结合 HuggingFace Accelerate 的分布式训练能力，并加入 ModelBundle 共享、按模块 checkpoint 控制等新特性。

## Feature Comparison / 功能对比

| Feature | MMEngine (MMLab) | HuggingFace Trainer | HF-Trainer (ours) |
|---|---|---|---|
| Config system | `.py` dict + registry | `TrainingArguments` dataclass | `.py` dict + registry (MMEngine) |
| Distributed | Custom FSDP (buggy) | Accelerate | Accelerate (native) |
| Per-module freeze/ckpt | Hardcoded in Trainer | Not supported | Config-driven `trainable`/`save_ckpt` |
| Multi-optimizer (GAN) | Limited | Not supported | Named optimizer dict, Trainer controls update order |
| Resume vs load_ckpt | Two separate APIs | `resume_from_checkpoint` | Unified `load_from` + `load_scope` |
| Model support | `BaseModel` only | `transformers` only | Any `nn.Module` / HF models |
| Diffusion support | No | No | First-class via `DiffusionTrainer` |
| Trainer/Pipeline sharing | No (duplicate code) | No pipeline concept | `ModelBundle` shares atomic forward functions |
| Evaluator/Visualizer input | `DataSample` (complex container) | dict (coupled to pipeline) | Plain dict, key convention only |
| Demo data | No | No | `data/{task}/demo/` + download script |
| Checkpoint IO | Custom, slow | Accelerate | Accelerate + selective save (skip frozen modules) |

## Key Differentiators / 核心差异

### Config-Driven Per-Module Control / 配置驱动的按模块控制

MMEngine and HuggingFace Trainer both lack fine-grained, config-driven control over which modules are trainable and which are saved in checkpoints. In HF-Trainer, this is declared per sub-module:

MMEngine 和 HuggingFace Trainer 都缺乏细粒度的、配置驱动的模块级训练和 checkpoint 控制。在 HF-Trainer 中，这是按子模块声明的：

```python
model = dict(
    type='WanModelBundle',
    text_encoder=dict(..., trainable=False, save_ckpt=False),
    transformer=dict(..., trainable=True, save_ckpt=True),
)
```

### ModelBundle Sharing / ModelBundle 共享

In traditional frameworks, Trainer and Pipeline duplicate forward logic (encode_text, predict_noise, etc.), which leads to inconsistencies. HF-Trainer's `ModelBundle` holds all atomic forward functions, and both Trainer and Pipeline call into it.

在传统框架中，Trainer 和 Pipeline 会重复实现前向逻辑（encode_text、predict_noise 等），容易产生不一致。HF-Trainer 的 `ModelBundle` 持有所有原子前向函数，Trainer 和 Pipeline 共同调用。

### Unified Checkpoint Loading / 统一的 Checkpoint 加载

Instead of separate `resume` and `load_checkpoint` APIs, HF-Trainer unifies them with `load_scope`:

HF-Trainer 不再区分 `resume` 和 `load_checkpoint`，而是统一用 `load_scope` 控制：

- `load_scope='model'` -- model weights only (transfer learning)
- `load_scope='full'` -- model + optimizer + scheduler + training meta (resume)
