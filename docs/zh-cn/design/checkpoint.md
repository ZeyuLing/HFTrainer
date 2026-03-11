# Checkpoint

HF-Trainer 只保留一个 checkpoint 概念，通过 `load_scope` 区分加载范围：

- `load_scope='model'`：只加载选择性的模型权重
- `load_scope='full'`：加载完整 accelerator state，用于 resume

## 文件组成

`model.pt`

- 只包含 `save_ckpt=True` 模块的嵌套 state dict
- 额外带有 `__hftrainer_meta__` 模块级元信息
- 当 `checkpoint_format='lora'` 时，LoRA 模块默认只保存 adapter 权重

Accelerate state 文件：

- `model.safetensors`
- `optimizer.bin`
- `scheduler.bin`
- `random_states_0.pkl`

`meta.pt`

- `global_step`
- `current_epoch`

## Resume 规则

- iter-based checkpoint 用“已完成 step 数”命名
- epoch-based checkpoint 用“已完成 epoch 数”命名
- `auto_resume=True` 会自动挑选 `work_dir` 中最新的 checkpoint

这样可以避免常见的 off-by-one 恢复问题，不会重复最后一步。

## LoRA 格式

- `checkpoint_format='lora'`：adapter-only checkpoint
- `checkpoint_format='full'`：完整的 LoRA 包装模块 state dict

当一个模块声明为 `trainable='lora'` 时，HF-Trainer 默认使用 `checkpoint_format='lora'`。
