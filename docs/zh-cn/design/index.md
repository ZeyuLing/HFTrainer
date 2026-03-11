# 设计总览

HF-Trainer 围绕几条核心原则展开：

- 用 MMEngine `Config` 与 `Registry` 做配置驱动构建
- 用 `accelerate` 作为运行时
- 直接使用 HuggingFace 原生组件
- 用 `ModelBundle` 共享任务逻辑

## 设计页面

- [ModelBundle](model_bundle.md)
- [Checkpoint](checkpoint.md)
- [Hooks](hooks.md)
- [LoRA](../lora.md)
- [多 Optimizer](multi_optimizer.md)
- [数据集](dataset.md)
- [评估与可视化](evaluation.md)
