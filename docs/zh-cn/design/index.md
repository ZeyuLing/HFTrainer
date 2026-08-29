# 设计总览

HFTrainer 围绕几条核心原则展开：

- 用 MMEngine `Config` 与 `Registry` 做配置驱动构建
- 用 `accelerate` 作为运行时
- 仓库自有模型组件，并且只通过本地 registry 解析
- 用 `ModelBundle` 共享训练与推理原子操作
- 各实现拥有自己的 artifact schema，并使用本地 LoRA
- 结构测试拒绝外部模型包 import

## 设计页面

- [ModelBundle](model_bundle.md)
- [Checkpoint](checkpoint.md)
- [Hooks](hooks.md)
- [LoRA](../lora.md)
- [多 Optimizer](multi_optimizer.md)
- [数据集](dataset.md)
- [评估与可视化](evaluation.md)
