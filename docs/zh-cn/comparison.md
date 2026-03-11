# 框架对比

HF-Trainer 处在 MMEngine 风格实验管理和 HuggingFace 原生运行时之间。

| 能力 | MMEngine 风格栈 | HF Trainer | HF-Trainer |
| --- | --- | --- | --- |
| 配置驱动实验 | 强 | 弱 | 强 |
| 原生 `accelerate` 运行时 | 否 | 是 | 是 |
| 训练/推理任务逻辑复用 | 弱 | 弱 | 通过 `ModelBundle` 强化 |
| 按模块 freeze/save 控制 | 零散 | 有限 | 配置驱动 |
| 多 optimizer | 可做但啰嗦 | 有限 | runner/trainer API 已支持 |
| 直接使用 diffusers / transformers | 往往要包一层 | 偏 transformers | 可直接使用 |

## 仍然不完整的部分

- GAN 和 DMD 现在已经有可运行的参考项目，但它们仍然是偏框架验证的 reference implementation，不是面向 benchmark 调优的复现结果。
- 当前公开文档以 `docs/en/` 和 `docs/zh-cn/` 为准。
