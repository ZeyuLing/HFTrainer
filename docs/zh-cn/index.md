# HF-Trainer 文档

HF-Trainer 是一个面向 HuggingFace 生态的配置驱动训练框架。当前代码库重点覆盖一组可运行任务栈，以及一组核心框架原语：`ModelBundle`、`Trainer`、`Pipeline`、`AccelerateRunner`、hooks、evaluators 和 visualizers。

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
- LTX-Video 2.5 蒸馏/Dev 推理与 managed LoRA 训练适配器

LTX 接口已针对固定官方源码版本完成合约测试；仓库 CI/测试环境没有执行 gated
22B 权重。
