# Memory and Precision

This page documents the config-level memory controls that HF-Trainer supports today.

## 1. Global Runtime Precision

Use the runner-level `accelerator` config for global AMP and accumulation:

```python
accelerator = dict(
    mixed_precision='bf16',          # 'no' | 'fp16' | 'bf16'
    gradient_accumulation_steps=4,
)
```

- `mixed_precision` controls the global `accelerate` AMP policy.
- `gradient_accumulation_steps` reduces per-step activation memory at the cost of throughput.

## 2. Per-Module Dtype

HF-Trainer supports two layers of per-module dtype control.

### 2.1 Pass Through To The Official HF Loader

If the underlying `transformers` / `diffusers` class already supports `torch_dtype` or `dtype`, keep using that loader API inside the sub-module config:

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

### 2.2 HF-Trainer Post-Load Cast

If you want a framework-level cast that works for any `nn.Module`, use `module_dtype`:

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

Accepted `module_dtype` values:

- `'fp32'`, `'float32'`, `'torch.float32'`
- `'fp16'`, `'float16'`, `'torch.float16'`
- `'bf16'`, `'bfloat16'`, `'torch.bfloat16'`
- a real `torch.dtype`

### Important Caveat

If you need a strict policy like `vae=fp32` and `transformer=bf16`, prefer:

- per-module `torch_dtype` / `module_dtype`
- `accelerator.mixed_precision='no'`

If global AMP is also enabled, `accelerate` may still autocast eligible ops on top of your module weights.

## 3. Gradient Checkpointing

Any bundle sub-module can now declare:

```python
transformer=dict(
    type='WanTransformer3DModel',
    from_pretrained=dict(pretrained_model_name_or_path=CKPT_PATH),
    gradient_checkpointing=True,
)
```

or:

```python
model=dict(
    type='AutoModelForCausalLM',
    from_pretrained=dict(pretrained_model_name_or_path=CKPT_PATH),
    gradient_checkpointing=dict(use_reentrant=False),
)
```

HF-Trainer enables checkpointing when the module exposes one of these hooks:

- `gradient_checkpointing_enable(...)`
- `enable_gradient_checkpointing(...)`

If the module does not expose a supported hook, HF-Trainer raises an explicit config error.

## 4. Other Memory-Saving Controls Already Supported

- `trainable=False`: freeze the module and avoid optimizer state for it.
- `trainable='lora'`: only train adapters instead of the full module.
- `checkpoint_format='lora'`: save adapter-only checkpoints by default for LoRA modules.
- `save_ckpt=False`: skip large frozen modules during save/load. This helps checkpoint IO and disk usage, not runtime GPU memory.
- optimizer choice is config-driven: you can switch to `Adafactor`, `SGD`, or other supported optimizers directly in config.

## 5. Model-Specific Loader Knobs

HF-Trainer also lets you pass through model-specific loading arguments under `from_pretrained`, for example:

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

These are passed directly to the underlying HF class. HF-Trainer does not normalize or validate every model-specific knob.

## 6. Not Yet Standardized By HF-Trainer

These memory features are still possible future framework work, but are not yet exposed as one unified config contract:

- xFormers / memory-efficient attention toggles
- attention slicing / VAE tiling style helpers
- 8-bit optimizer presets
- module-level autocast disable / force-fp32 islands
- packaged ZeRO/FSDP offload presets beyond raw `accelerate` usage

For now, use HF-native pass-through kwargs where available, and keep the standardized HF-Trainer contract to:

- `accelerator.mixed_precision`
- `accelerator.gradient_accumulation_steps`
- `from_pretrained.torch_dtype` / `dtype`
- `module_dtype`
- `gradient_checkpointing`
- `trainable` / `save_ckpt` / `checkpoint_format`
