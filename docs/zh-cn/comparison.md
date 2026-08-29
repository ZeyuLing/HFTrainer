# 框架对比

HFTrainer 把配置驱动实验管理与仓库自有模型执行结合起来。它的核心区别是代码
所有权，而不是在当前环境恰好安装的模型包外再套一层 wrapper。

| 能力 | 脚本型项目 | 外部模型包 wrapper | HFTrainer |
| --- | --- | --- | --- |
| 配置驱动构建 | 项目自定义 | 视 wrapper 而定 | MMEngine config + 本地 registry |
| 分布式运行时 | 项目自定义 | 通常由对应框架决定 | Accelerate |
| 模型数学所有权 | 复制或隐含 | 委托给已安装包 | 本地 `hftrainer/models/<implementation>/network` |
| 训练/推理复用 | 经常重复实现 | 取决于 wrapper | 共享 `ModelBundle` 原子操作 |
| Artifact schema | 零散 | 受包版本控制 | 每个实现自行拥有并校验 |
| LoRA | 自定义或委托 | 依赖 adapter 包 | 本地 `LoRALinear` 实现 |
| 组件解析 | import 与动态路径 | 外部包 class lookup | 仅本地 `MODEL_COMPONENTS` |
| 依赖漂移 | 手工处理 | 模型行为可能随包版本变化 | 测试禁止模型包 import |
| 多 optimizer 训练 | 项目自定义 | 通常有限 | runner/trainer 协议 |

通用基础设施依赖仍然是有意保留的：PyTorch 提供 tensor kernel，Accelerate 负责
分布式编排，MMEngine 提供 config/registry 原语，safetensors、NumPy、Pillow
支持 artifact 与数据处理；它们不提供 HFTrainer 的模型定义或任务算法。

## 当前验证边界

- ViT、LLaMA、SD1.5、DMD、StyleGAN2 与 Wan 都有降规模本地模型测试；具体
  checkpoint 限制以各实现的 config/文档为准。
- LTX 的 model/trainer/pipeline 源码来自一个固定并经过修改的本地快照，继续受其
  自有许可证约束。合约测试与 tiny 本地 Gemma 路径已经通过，但仓库测试环境没有
  端到端执行 gated 22B 工作流。
- reference implementation 用于证明框架结构，本身不等同于 benchmark 复现声明。
