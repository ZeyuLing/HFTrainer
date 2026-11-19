# 多 Optimizer

## 当前范围

HF-Trainer 在 runner 和 trainer API 层已经支持多 optimizer，并且现在已经补上了可运行的 StyleGAN2 风格 GAN 参考项目和 DMD 风格蒸馏参考项目。

已经具备的框架能力：

- 命名 `optimizer` dict
- 命名 `lr_scheduler` dict
- `BaseTrainer.trainer_controls_optimization = True`
- 通过 `set_optimizers()` 注入 optimizer / scheduler
- 判别器调度控制：
  - `disc_start_step`
  - `disc_warmup_steps`
  - `disc_update_interval`

已存在的参考 trainer：

- `StyleGAN2Trainer`
- `DMDTrainer`

## 判别器调度

`StyleGAN2Trainer` 和 `DMDTrainer` 现在都支持延迟启用判别器。

可用 trainer 字段：

- `disc_start_step`：在完成多少个训练 step 之后再启用对抗分支
- `disc_warmup_steps`：从 `0` 线性爬升到目标对抗权重
- `disc_update_interval`：判别器启用后的更新频率

语义说明：

- 调度基于“已完成的全局 step”计算，所以 `disc_start_step=5000` 表示判别器会从第 5001 个 step 开始生效
- `disc_update_interval` 从判别器首次激活那一步开始计数，因此 delayed-start 不会把更新相位错开

## 预期用法

当 trainer 需要交替更新多个模块时，runner 不应再走默认的单 loss 优化路径。此时 trainer 需要自己执行：

- `self.accelerator.backward(loss)`
- `optimizer.step()`
- `optimizer.zero_grad()`
- 必要时 scheduler.step()

并返回 `{'loss': None, ...}`，避免 runner 重复 step。

## 当前限制

- GAN 和 DMD 目前仍然是偏框架验证的 reference implementation，不是已经调到 benchmark 水平的复现结果。
- DMD 支持读取预计算 regression pairs，但 demo config 也提供了在线 teacher target 生成路径，方便直接跑通。

## 优化器类与 Muon

`optimizer.type` 可填写 PyTorch 的类名（如 `AdamW`），也可填写已安装的
`torch.optim.Optimizer` 子类的完整导入路径。依赖仅在选择该优化器时导入。

```python
optimizer = dict(
    type='transformers.optimization.Adafactor',
    lr=1e-4,
    relative_step=False,
    scale_parameter=False,
)
```

`type='Muon'` 使用 PyTorch 2.9+ 的 `torch.optim.Muon`。
Runner 对二维隐藏层参数应用 Muon，对 embedding、输出头、bias 和其他参数
应用 AdamW。可用 `exclude` 添加参数名正则表达式，将匹配的参数交给 AdamW。
`adamw` 可独立覆盖学习率、权重衰减和 AdamW 参数。
`adjust_lr_fn='match_rms_adamw'` 选用 PyTorch 的 Moonshot 学习率缩放。

LTX-Video 的示例调用如下，学习率是配置示例。

```bash
python tools/train.py configs/ltx_video/train_ltx_video_2_5_lora.py --cfg-options \
  optimizer.type=Muon optimizer.lr=4e-4 optimizer.adamw.lr=4e-5 \
  optimizer.adjust_lr_fn=match_rms_adamw
```

自动分组支持由 Runner 控制优化的单设备和 DDP 训练。FSDP、DeepSpeed
分片训练需要对应后端提供兼容的算法。更换优化器时用
`--load-from PATH --load-scope model` 加载已有权重并重新初始化优化器。
