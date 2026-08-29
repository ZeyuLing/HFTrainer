# HFTrainer 文档

HFTrainer 是模型实现归当前仓库所有的配置驱动训练与推理框架。`ModelBundle`、
trainer 与 pipeline 共享同一份本地组件图；Accelerate 负责分布式运行时编排，
不提供模型定义。

## 建议阅读顺序

- [安装说明](installation.md)
- [快速开始](quickstart.md)
- [LTX-Video 2.5](models/ltx_video_2_5.md)
- [模型接入](integration.md)
- [API 参考](api_reference.md)
- [显存与精度](memory.md)
- [LoRA](lora.md)
- [架构设计](architecture.md)
- [分布式训练](distributed.md)
- [实验目录](experiment_dir.md)
- [任务矩阵](tasks.md)
- [框架对比](comparison.md)

## 设计文档

- [设计总览](design/index.md)
- [ModelBundle](design/model_bundle.md)
- [Checkpoint](design/checkpoint.md)
- [Hooks](design/hooks.md)
- [多 Optimizer](design/multi_optimizer.md)
- [数据集](design/dataset.md)
- [评估与可视化](design/evaluation.md)

## 当前状态

已打通 demo：

- 图像分类
- 文生图
- Causal LM SFT
- Causal LM LoRA
- 文生视频
- StyleGAN2 风格 GAN 训练
- DMD 风格蒸馏
- LTX-Video 2.5 蒸馏/Dev 推理与本地 managed LoRA 训练

随包 LTX 实现是固定到一个源码 revision 的修改版快照。本地 config/API 合约和
tiny Gemma 路径已经测试；仓库测试环境没有执行 gated 22B 工作流。
