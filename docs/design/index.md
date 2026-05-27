# Design Overview / 设计概述

This section documents the design rationale behind HF-Trainer. Each page is bilingual (English and Chinese).

本节记录 HF-Trainer 的设计理念。每一页均为双语（英文和中文）。

## Motivation / 动机

MMLab (MMDet, MMPose, MMEngine) defined an elegant training framework paradigm: declaratively instantiate all modules through `.py` config files with dict syntax, and launch training with a single `dist_train.sh`. This design excels at modularity and reusability.

MMLab (MMDet, MMPose, MMEngine) 定义了一套非常优雅的训练框架范式：通过 `.py` config 文件以 dict 形式声明式地实例化所有模块，通过 `dist_train.sh` 一键启动训练。这种设计在模块化和可复用性上做得很好。

However, in practice, several serious problems emerged:

但在实际使用中，暴露出了若干严重问题：

### 1. Outdated Infrastructure, Disconnected from HuggingFace Ecosystem / 基建陈旧，与 HuggingFace 生态脱节

- Poor quality FSDP / DeepSpeed (ZeRO) implementations with many bugs
- Cannot use `accelerate` to uniformly manage training (device placement, gradient accumulation, mixed precision, etc.)
- Checkpoint save/load doesn't follow `accelerator.save_state()` / `accelerator.load_state()`, leading to extremely slow checkpoint I/O for large models

- FSDP / DeepSpeed (ZeRO) 等分布式训练的实现质量差、Bug 多
- 不能通过 `accelerate` 统一管理训练过程（device placement, gradient accumulation, mixed precision 等）
- Checkpoint 存储/加载不遵循 `accelerator.save_state()` / `accelerator.load_state()`，导致大模型 checkpoint 读写极慢

### 2. Over-Abstraction, Steep Learning Curve / 过度封装，学习曲线陡峭

- `DataSample`, `BaseModel.forward(mode='loss'|'predict'|'tensor')` and similar abstractions confuse beginners
- Datasets must inherit `BaseDataset` and follow specific `__getitem__` return formats
- Evaluation is tied to `BaseMetric`, incompatible with HuggingFace `evaluate`

- `DataSample`, `BaseModel.forward(mode='loss'|'predict'|'tensor')` 等抽象让初学者困惑
- Dataset 需要继承 `BaseDataset` 并遵循特定的 `__getitem__` 返回格式
- 评估流程绑定在 `BaseMetric` 上，与 HuggingFace `evaluate` 库不兼容

### 3. Not Interoperable with diffusers / transformers / 与 diffusers / transformers 不互通

- Cannot directly use `diffusers.StableDiffusionPipeline`, `transformers.AutoModelForCausalLM`, etc.
- Scheduler, Tokenizer, Processor etc. need extra wrappers to work in the MMEngine system

- 无法直接使用 `diffusers.StableDiffusionPipeline`, `transformers.AutoModelForCausalLM` 等
- Scheduler, Tokenizer, Processor 等组件需要额外包装才能在 MMEngine 体系中使用

### 4. Trainer and Pipeline Code Duplication / Trainer 与 Pipeline 代码重复

- After writing the Trainer, rewriting the inference Pipeline requires duplicating a lot of forward computation logic (encode text, denoise loop, etc.), easily leading to inconsistencies

- 训练写完 Trainer 后，改写推理 Pipeline 时需要重复大量前向计算逻辑（encode text、denoise loop 等），两处容易出现不一致

## Goal / 目标

HF-Trainer's goal: retain MMLab's config-driven declarative design paradigm, but migrate the runtime entirely to the HuggingFace Accelerate ecosystem, and use the `ModelBundle` mechanism to completely solve the Trainer/Pipeline code duplication problem.

HF-Trainer 的目标：保留 MMLab config-driven 的声明式设计范式，但将运行时完全迁移到 HuggingFace Accelerate 生态上，并通过 `ModelBundle` 机制彻底解决 Trainer/Pipeline 代码重复问题。

---

## Design Principles / 设计原则

1. **Config-Driven, Registry-Based**: Retain MMEngine's `Config` + `Registry` mechanism / 保留 MMEngine 的 `Config` + `Registry` 机制
2. **Accelerate-Native**: All distributed, mixed precision, gradient accumulation logic is managed by `Accelerator` / 所有分布式、混合精度、gradient accumulation 等逻辑均由 `Accelerator` 管理
3. **HuggingFace-First**: Directly use diffusers / transformers native classes, no unnecessary wrappers / 直接使用 diffusers / transformers 原生类，不做多余包装
4. **Per-Module Control**: Each sub-module independently controls trainable, checkpoint load/save / 每个子模块独立控制 trainable、checkpoint load/save
5. **ModelBundle = Shared Core**: Trainer and Pipeline share the same `ModelBundle`, all forward computation logic is written once / Trainer 和 Pipeline 共享同一个 `ModelBundle`，所有前向计算逻辑只写一次

---

## Design Documents / 设计文档

- [ModelBundle](model_bundle.md) -- Per-module control + Trainer/Pipeline sharing / 按模块控制 + Trainer/Pipeline 共享
- [Checkpoint](checkpoint.md) -- Unified `load_from` / `load_scope` / 统一的 `load_from` / `load_scope`
- [Multi-Optimizer](multi_optimizer.md) -- GAN multi-optimizer support / GAN 多 optimizer 支持
- [Dataset](dataset.md) -- Dataset directory structure / 数据集目录结构
- [Evaluation](evaluation.md) -- Evaluator/Visualizer dict interface / Evaluator/Visualizer dict 接口
