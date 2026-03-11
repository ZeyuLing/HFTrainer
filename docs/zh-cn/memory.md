# 显存与精度

这一页整理 HF-Trainer 当前已经支持的、可通过 config 控制的显存与精度策略。

## 1. 全局运行时精度

用 runner 级别的 `accelerator` 配置全局 AMP 和梯度累积：

```python
accelerator = dict(
    mixed_precision='bf16',          # 'no' | 'fp16' | 'bf16'
    gradient_accumulation_steps=4,
)
```

- `mixed_precision` 控制全局 `accelerate` AMP 策略。
- `gradient_accumulation_steps` 通过减少单步 activation 占用来换吞吐。

## 2. 按模块控制 dtype

HF-Trainer 现在支持两层按模块 dtype 控制。

### 2.1 透传给官方 HF Loader

如果底层 `transformers` / `diffusers` 类本来就支持 `torch_dtype` 或 `dtype`，优先继续沿用官方 loader 接口：

```python
model = dict(
    type='SD15Bundle',
    text_encoder=dict(
        type='CLIPTextModel',
        from_pretrained=dict(
            pretrained_model_name_or_path=CKPT_PATH,
            subfolder='text_encoder',
            torch_dtype='bf16',
        ),
        trainable=False,
        save_ckpt=False,
    ),
)
```

### 2.2 HF-Trainer 的 Post-Load Cast

如果你希望框架在模块构建完成后统一 cast，或者底层 loader 本身没有合适的 dtype 参数，就用 `module_dtype`：

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

- `'fp32'`、`'float32'`、`'torch.float32'`
- `'fp16'`、`'float16'`、`'torch.float16'`
- `'bf16'`、`'bfloat16'`、`'torch.bfloat16'`
- 真实的 `torch.dtype`

### 重要注意事项

如果你需要严格的“`vae=fp32`、`transformer=bf16`”策略，建议组合是：

- 按模块设置 `torch_dtype` / `module_dtype`
- `accelerator.mixed_precision='no'`

如果同时开启全局 AMP，`accelerate` 仍然可能在 eligible op 上继续 autocast。

## 3. Gradient Checkpointing

现在任意 bundle 子模块都可以直接声明：

```python
transformer=dict(
    type='WanTransformer3DModel',
    from_pretrained=dict(pretrained_model_name_or_path=CKPT_PATH),
    gradient_checkpointing=True,
)
```

或者：

```python
model=dict(
    type='AutoModelForCausalLM',
    from_pretrained=dict(pretrained_model_name_or_path=CKPT_PATH),
    gradient_checkpointing=dict(use_reentrant=False),
)
```

HF-Trainer 会在模块暴露下列 hook 时自动接通：

- `gradient_checkpointing_enable(...)`
- `enable_gradient_checkpointing(...)`

如果模块没有兼容 hook，会直接抛出明确的配置错误。

## 4. 已支持的其他省显存手段

- `trainable=False`：冻结模块，不为它创建 optimizer state。
- `trainable='lora'`：只训练 adapter，而不是整个模块。
- `checkpoint_format='lora'`：LoRA 模块默认只存 adapter checkpoint。
- `save_ckpt=False`：保存/恢复时跳过大模块。这个主要节省 checkpoint IO 和磁盘，不直接减少运行时 GPU 显存。
- optimizer 也是 config 驱动的：可以直接切到 `Adafactor`、`SGD` 等较轻量方案。

## 5. 模型专属 Loader 参数

HF-Trainer 也支持在 `from_pretrained` 下直接透传模型专属参数，例如：

```python
model=dict(
    type='AutoModelForCausalLM',
    from_pretrained=dict(
        pretrained_model_name_or_path=CKPT_PATH,
        torch_dtype='bf16',
        low_cpu_mem_usage=True,
        attn_implementation='flash_attention_2',
    ),
)
```

这些参数会直接传给底层 HF 类。HF-Trainer 不会统一规范或校验每一个模型专属开关。

## 6. 还没有统一成框架契约的能力

下面这些可以作为后续演进方向，但目前还没有做成统一配置接口：

- xFormers / memory-efficient attention 开关
- attention slicing / VAE tiling 这类 helper
- 8-bit optimizer 预设
- 按模块关闭 autocast / 强制 fp32 island
- 除原生 `accelerate` 用法外的 ZeRO/FSDP offload 预设

当前已经标准化的 HF-Trainer 显存契约是：

- `accelerator.mixed_precision`
- `accelerator.gradient_accumulation_steps`
- `from_pretrained.torch_dtype` / `dtype`
- `module_dtype`
- `gradient_checkpointing`
- `trainable` / `save_ckpt` / `checkpoint_format`
