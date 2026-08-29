# 显存与精度

本页说明 HFTrainer 已实现的 config 级显存与精度控制。模型专属加载参数由仓库
本地 component class 解释，不会被透传给外部模型 loader。

## 1. 全局运行时精度

用 runner 级 `accelerator` 配置自动混合精度与梯度累积：

```python
accelerator = dict(
    mixed_precision='bf16',  # 'no' | 'fp16' | 'bf16'
    gradient_accumulation_steps=4,
)
```

- `mixed_precision` 选择 Accelerate 的 AMP 策略。
- `gradient_accumulation_steps` 减少每次 optimizer step 的 activation 峰值，
  通常会牺牲部分吞吐。

Accelerate 是运行时基础设施，不提供 HFTrainer 的模型数学、tokenizer、scheduler、
trainer 或 pipeline 实现。

## 2. 按模块控制 dtype

HFTrainer 提供两层本地 dtype 控制。

### 2.1 Artifact loader 的 dtype

本地 component loader 可以在自己的 `from_pretrained` 合约中声明
`torch_dtype` 或 `dtype`。例如仓库自有的 Wan encoder：

```python
text_encoder=dict(
    type='UMT5EncoderModel',
    from_pretrained=dict(
        pretrained_model_name_or_path=CKPT_PATH + '/text_encoder',
        torch_dtype='bf16',
    ),
    trainable=False,
    save_ckpt=False,
)
```

这个类名通过 `MODEL_COMPONENTS` 解析到 `hftrainer.models.wan` 下的代码；
`from_pretrained` 描述磁盘 artifact，不表示选择外部实现。

### 2.2 Bundle 的 post-load cast

使用 `module_dtype` 可以在本地组件构建完成后统一调用
`nn.Module.to(dtype=...)`：

```python
model = dict(
    type='SD15Bundle',
    vae=dict(
        type='AutoencoderKL',
        from_pretrained=dict(
            pretrained_model_name_or_path=CKPT_PATH,
            subfolder='vae',
        ),
        module_dtype='fp32',
        trainable=False,
        save_ckpt=False,
    ),
    unet=dict(
        type='UNet2DConditionModel',
        from_pretrained=dict(
            pretrained_model_name_or_path=CKPT_PATH,
            subfolder='unet',
        ),
        module_dtype='bf16',
        trainable=True,
        save_ckpt=True,
    ),
)
```

`module_dtype` 支持：

- `'fp32'`、`'float32'`、`'torch.float32'`；
- `'fp16'`、`'float16'`、`'torch.float16'`；
- `'bf16'`、`'bfloat16'`、`'torch.bfloat16'`；
- 真实的 `torch.dtype`。

如果需要严格的 `vae=fp32`、`transformer=bf16`，请逐模块设置 dtype，并使用
`accelerator.mixed_precision='no'`。否则全局 AMP 仍可能 autocast 合适的算子。

## 3. Gradient checkpointing

任意 bundle 子模块都可以请求 activation checkpointing：

```python
transformer=dict(
    type='WanTransformer3DModel',
    from_pretrained=dict(
        pretrained_model_name_or_path=CKPT_PATH + '/transformer',
    ),
    gradient_checkpointing=True,
)
```

`ModelBundle` 会调用本地模块的 `gradient_checkpointing_enable(...)` 或
`enable_gradient_checkpointing(...)`。只有本地 hook 明确记录了关键字参数时，
才应传 dict。两者都不存在，或已记录参数不被接受时，会直接抛出明确的配置错误。

## 4. 其他已支持控制

- `trainable=False`：冻结模块，不为其创建 optimizer state。
- `trainable='lora'`：注入 HFTrainer 本地低秩层，只训练 adapter；详见
  [LoRA](lora.md)。
- `checkpoint_format='lora'`：只保存 adapter checkpoint。
- `save_ckpt=False`：选择性保存/加载时跳过冻结模块；它减少 checkpoint I/O 与
  磁盘占用，不直接减少运行时显存。
- optimizer 只从 `torch.optim` 解析。HFTrainer 命名 schedule 使用
  `hftrainer.optim.schedulers`；显式 PyTorch scheduler class 可以从
  `torch.optim.lr_scheduler` 解析。

HFTrainer 尚未拥有经过验证的本地 4-bit linear kernel，因此不开放 QLoRA。

## 5. 模型专属参数

只传入当前仓库本地模型明确记录的 loader 参数。HFTrainer 有意不提供把任意参数
透传给另一模型框架的通用逃逸入口；不支持的参数会在本地 constructor/loader
边界报错。

## 6. 尚未统一的能力

以下能力目前还不是跨模型统一 config 合约：

- memory-efficient attention backend；
- attention slicing 与 VAE tiling helper；
- 8-bit optimizer preset；
- 按模块关闭 autocast 或强制 fp32 island；
- Accelerate 配置之外的打包 ZeRO/FSDP offload preset。

当前稳定的跨模型控制包括：

- `accelerator.mixed_precision`；
- `accelerator.gradient_accumulation_steps`；
- 各本地实现明确支持的 `from_pretrained.torch_dtype` / `dtype`；
- `module_dtype`；
- `gradient_checkpointing`；
- `trainable`、`save_ckpt` 与 `checkpoint_format`。
