# API 参考

这一页记录 HF-Trainer 面向用户的公共 API。重点覆盖用户会直接实例化、继承或调用的类、方法和 CLI 入口。

它不会尝试把仓库里的每个内部 helper 都逐个列出来。各任务 bundle 内部更细的原子前向函数，仍然建议结合对应源码阅读。

## 覆盖范围

这一页覆盖：

- 框架主入口：`AccelerateRunner`、`ModelBundle`、`BaseTrainer`、`BasePipeline`
- validation / runtime 扩展点：hooks、evaluators、visualizers
- checkpoint 工具：`find_latest_checkpoint`、`load_checkpoint`、`save_checkpoint`
- 命令行入口：`tools/train.py`、`tools/infer.py`

可运行任务栈见 [任务矩阵](tasks.md)。

## `AccelerateRunner`

导入方式：

```python
from hftrainer import AccelerateRunner
```

典型用法：

```python
runner = AccelerateRunner.from_cfg(cfg)
runner.train()
```

### 主要 API

| 方法 | 作用 | 输入 |
| --- | --- | --- |
| `from_cfg(cfg)` | 从 config 构建完整运行时 | `cfg`: `dict` 或 `mmengine.Config` |
| `train()` | 执行训练循环 | 无 |
| `val()` | 执行一次 validation | 无 |
| `save_checkpoint()` | 在 `work_dir` 下保存 checkpoint | 无 |

### `from_cfg(cfg)` 需要的顶层配置

| 配置项 | 是否必须 | 作用 |
| --- | --- | --- |
| `model` | 是 | bundle 配置 |
| `trainer` | 是 | trainer 配置 |
| `train_dataloader` | 训练时必须 | 训练数据 / dataloader 配置 |
| `optimizer` | 训练时必须 | optimizer 配置，支持单个或命名 dict |
| `train_cfg` | 推荐 | iter/epoch 循环设置 |
| `lr_scheduler` | 可选 | scheduler 配置 |
| `default_hooks` | 可选 | hook 配置 |
| `val_dataloader` | 可选 | 验证 dataloader 配置 |
| `val_evaluator` | 可选 | evaluator 配置 |
| `val_visualizer` | 可选 | visualizer 配置 |
| `accelerator` | 可选 | accelerate 运行时配置 |
| `work_dir` | 可选 | 实验根目录 |
| `auto_resume` | 可选 | 从 `work_dir` 自动恢复最新 checkpoint |
| `load_from` | 可选 | 显式 checkpoint 加载配置 |

### 重要属性

| 属性 | 含义 |
| --- | --- |
| `bundle` | 已实例化的 `ModelBundle` |
| `trainer` | 已实例化的 trainer |
| `hooks` | 已构建并按优先级排序的 hooks |
| `evaluators` | validation 指标组件 |
| `visualizers` | validation 可视化组件 |
| `optimizers` | 命名 optimizer dict |
| `lr_schedulers` | 命名 scheduler dict |
| `global_step` | 已完成的 optimizer step 数 |
| `current_epoch` | 已完成的 epoch 数 |
| `work_dir` | checkpoint 根目录 |
| `run_dir` | 带时间戳的日志 / config / tensorboard 目录 |

## `ModelBundle`

导入方式：

```python
from hftrainer.models import ModelBundle
```

作用：

- 持有任务所需的全部子模块
- 从 config 实例化这些子模块
- 控制按模块的 trainable / checkpoint 语义
- 为 trainer 和 pipeline 提供共享的原子前向函数

### 子模块配置键

每个 bundle 子模块都可以声明：

| 键 | 类型 | 含义 |
| --- | --- | --- |
| `type` | `str` | registry key / 类名 |
| `from_pretrained` | `dict` | HuggingFace 风格 pretrained 加载参数 |
| `from_config` | `dict` | 按配置构建的参数 |
| `from_single_file` | `dict` | 单文件加载参数 |
| `trainable` | `bool` 或 `'lora'` | 全量训练、冻结，或注入 LoRA |
| `save_ckpt` | `bool` | 是否参与选择性 checkpoint save/load |
| `checkpoint_format` | `'full'` 或 `'lora'` | 保存全量权重或 adapter-only 权重 |
| `lora_cfg` | `dict` | `trainable='lora'` 时的 PEFT LoRA 配置 |
| `gradient_checkpointing` | `bool` 或 `dict` | 对暴露兼容 HF hook 的模块开启 activation checkpointing |
| `module_dtype` | dtype 字符串或 `torch.dtype` | 模块构建完成后施加的 dtype cast |

### 主要 API

| 方法 | 作用 | 输入 |
| --- | --- | --- |
| `from_config(cfg=None, **kwargs)` | 通用父类 bundle 构造入口 | config dict 或 `mmengine.ConfigDict` |
| `from_pretrained(path, config_overrides=None, **kwargs)` | 通用 HuggingFace 风格 bundle 构造入口 | pretrained 路径和任务相关参数 |
| `_build_modules(modules_cfg)` | 实例化并配置子模块 | `modules_cfg: dict` |
| `trainable_parameters()` | 返回所有可训练参数 | 无 |
| `trainable_named_parameters()` | 产生可训练 `(name, param)` 对 | 无 |
| `get_module_parameters(*names)` | 按子模块名收集参数 | 模块名列表 |
| `get_module_build_cfg(name)` | 返回某个子模块归一化后的构建配置 | 子模块名 |
| `get_module_pretrained_path(name)` | 返回某个子模块记录下来的 `from_pretrained` 路径 | 子模块名 |
| `state_dict_to_save()` | 构建选择性保存 dict | 无 |
| `load_state_dict_selective(state_dict, strict=False)` | 只加载 checkpoint 中出现的模块 | nested 或 flat state dict |
| `checkpoint_metadata()` | 描述保存模块及其 format | 无 |
| `merge_lora_weights(module_names=None)` | 把 adapter 合并进 base module | 可选模块名列表 |
| `save_pretrained(save_directory, **kwargs)` | 导出任务相关的推理 artifact | 导出目录 |

### 构造语义

- `from_config(...)` 是完全通用的，应该作为自研模型的默认入口。
- `from_pretrained(...)` 是父类上的统一 public API，但每个 HF-native bundle 仍然要通过 `_bundle_config_from_pretrained(...)` 定义“一个 pretrained artifact 如何映射成多个子模块”。
- `save_pretrained(...)` 保持任务相关，只有当 bundle 能导出官方推理 API 可以读取的 artifact 时才应该实现。
- `from_pretrained.torch_dtype` / `dtype` 会直接透传给底层 HF loader，`module_dtype` 则是 HF-Trainer 自己的 post-load cast。
- 全局 AMP、按模块 dtype 和 gradient checkpointing 的推荐写法见 [显存与精度](memory.md)。

### `load_state_dict_selective(...)`

可接受的输入格式：

- nested 保存格式：`{module_name: module_state_dict}`
- 可按模块前缀拆分的 flat checkpoint dict
- `checkpoint_format='lora'` 写出的 adapter-only checkpoint

### `merge_lora_weights(...)`

当你希望导出或推理前把 LoRA adapter 合并进 base weight 时调用：

```python
bundle.merge_lora_weights()
```

如果 `module_names=None`，会合并 bundle 里全部 LoRA 子模块。

## `BaseTrainer`

导入方式：

```python
from hftrainer import BaseTrainer
```

它是任务训练逻辑的基类。

### 主要 API

| 方法 | 作用 | 输入 |
| --- | --- | --- |
| `train_step(batch)` | 执行一次训练步 | dataloader batch dict |
| `val_step(batch)` | 执行一次验证步 | dataloader batch dict |
| `set_optimizers(optimizers, lr_schedulers=None)` | 注入命名 optimizers / schedulers | runner 构建好的 dict |
| `get_optimizer(name)` | 取命名 optimizer | optimizer 名字 |
| `get_lr_scheduler(name)` | 取命名 scheduler | scheduler 名字 |
| `get_global_step()` | 当前 iter 开始前已经完成的 step 数 | 无 |
| `get_current_step()` | 当前 1-based step 序号 | 无 |
| `get_discriminator_factor(...)` | delayed-start / warmup 权重 helper | 权重和调度参数 |
| `should_update_discriminator(...)` | 判别器更新 cadence helper | 起始步和间隔参数 |

### 多 optimizer 协议

设置：

```python
trainer_controls_optimization = True
```

开启后：

- runner 会把 optimizers 和 schedulers 注入进来
- runner 会跳过默认的 backward / step / zero_grad 路径
- `train_step()` 需要自己调用 `self.accelerator.backward(...)` 并推进 optimizer
- `train_step()` 应返回 `{'loss': None, ...}` 再附加日志项

## `BasePipeline`

导入方式：

```python
from hftrainer.pipelines import BasePipeline
```

Pipeline 负责组装推理阶段控制流，并始终消费一个 bundle。

### 主要 API

| 方法 | 作用 | 输入 |
| --- | --- | --- |
| `__init__(bundle, **kwargs)` | 从 bundle 构造 pipeline | 任务 bundle |
| `from_config(bundle_cls, bundle_cfg, **kwargs)` | 从 config 构建 bundle 并包装成 pipeline | bundle 类、bundle 配置 |
| `from_pretrained(bundle_cls, path, bundle_kwargs=None, **kwargs)` | 从 pretrained artifact 构建 bundle 并包装成 pipeline | bundle 类、pretrained 路径 |
| `from_checkpoint(bundle_cls, bundle_cfg, ckpt_path, **kwargs)` | 构建 bundle 并加载 checkpoint | bundle 类、bundle 配置、checkpoint 路径 |
| `__call__(...)` | 执行推理 | 任务相关参数 |

## Hooks

内置 hook 的详细文档见 [Hook 系统](design/hooks.md)。目前公开的内置 hook 有：

| Hook | 作用 | 关键参数 |
| --- | --- | --- |
| `LoggerHook` | 打印并记录标量指标 | `interval`, `by_epoch` |
| `CheckpointHook` | 保存周期 checkpoint 和最终 checkpoint | `interval`, `max_keep_ckpts`, `save_last`, `by_epoch` |
| `EMAHook` | 维护可训练模块的 EMA 副本 | `decay`, `update_interval` |
| `LRSchedulerHook` | 兼容占位 | `by_epoch` |

## Evaluator 与 Visualizer

### `BaseEvaluator`

导入方式：

```python
from hftrainer.evaluation import BaseEvaluator
```

| 方法 | 作用 |
| --- | --- |
| `reset()` | 清空 validation 累积结果 |
| `process(output)` | 消费一个 `val_step()` 输出 dict |
| `compute()` | 返回最终指标 dict |
| `compute_from_outputs(outputs)` | 面向完整输出列表的便捷 helper |

### `BaseVisualizer`

导入方式：

```python
from hftrainer.visualization import BaseVisualizer
```

| 方法 | 作用 |
| --- | --- |
| `visualize(output, step)` | 展示一个 validation 输出 dict |
| `should_visualize(step)` | 基于间隔的 gating helper |

## Checkpoint 工具函数

导入方式：

```python
from hftrainer.utils.checkpoint_utils import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
```

| 函数 | 作用 | 输入 |
| --- | --- | --- |
| `find_latest_checkpoint(work_dir)` | 查找最新 checkpoint 目录 | 实验根目录 |
| `load_checkpoint(path, map_location='cpu')` | 加载单文件或 checkpoint 目录 | 文件路径或 checkpoint 目录 |
| `save_checkpoint(state_dict, path, use_safetensors=True)` | 把 state dict 写到磁盘 | state dict 和输出路径 |

## CLI 参考

### `tools/train.py`

用法：

```bash
python3 tools/train.py CONFIG.py [OPTIONS]
```

| 参数 | 含义 |
| --- | --- |
| `config` | config 文件路径 |
| `--work-dir` | 覆盖 `cfg.work_dir` |
| `--auto-resume` | 从 `work_dir` 中最新 checkpoint 自动恢复 |
| `--load-from` | 显式 checkpoint 路径 |
| `--load-scope` | `model` 或 `full` |
| `--cfg-options` | 从命令行覆盖配置项 |

### `tools/infer.py`

用法：

```bash
python3 tools/infer.py --config CONFIG.py --checkpoint CKPT_DIR [OPTIONS]
```

| 参数 | 含义 |
| --- | --- |
| `--config` | config 文件路径 |
| `--checkpoint` | checkpoint 目录或 checkpoint 文件 |
| `--prompt` | 生成类任务的文本 prompt |
| `--input` | 分类类任务的输入文件 |
| `--output` | 输出图片 / 视频路径 |
| `--num-steps` | diffusion 去噪步数 |
| `--num-samples` | 无条件生成样本数 |
| `--num-frames` | 视频帧数 |
| `--max-new-tokens` | LLM 生成长度 |
| `--height`, `--width` | 输出空间尺寸 |
| `--merge-lora` | 推理前把 LoRA adapter 合并进 base weight |
| `--device` | `cuda` 或 `cpu` |

## 任务类映射

| 任务 | Bundle | Trainer | Pipeline |
| --- | --- | --- | --- |
| Classification | `ViTBundle` | `ClassificationTrainer` | `ClassificationPipeline` |
| Text-to-image | `SD15Bundle` | `SD15Trainer` | `SD15Pipeline` |
| Causal LM | `CausalLMBundle` | `CausalLMTrainer` | `CausalLMPipeline` |
| Text-to-video | `WanBundle` | `WanTrainer` | `WanPipeline` |
| GAN | `StyleGAN2Bundle` | `GANTrainer` | `StyleGAN2Pipeline` |
| DMD | `DMDBundle` | `DMDTrainer` | `DMDPipeline` |

可运行 config 和任务相关说明见 [任务矩阵](tasks.md)。
