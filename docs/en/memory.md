# Memory and Precision

This page documents the config-level memory controls implemented by HFTrainer.
Model-specific loading options are interpreted by repository-local component
classes; they are not forwarded to an external model loader.

## 1. Global runtime precision

Use the runner-level `accelerator` config for automatic mixed precision and
gradient accumulation:

```python
accelerator = dict(
    mixed_precision='bf16',  # 'no' | 'fp16' | 'bf16'
    gradient_accumulation_steps=4,
)
```

- `mixed_precision` selects the Accelerate AMP policy.
- `gradient_accumulation_steps` reduces the activation footprint per optimizer
  step, usually at a throughput cost.

Accelerate is runtime infrastructure. It does not provide HFTrainer's model
math, tokenizer, scheduler, trainer, or pipeline implementation.

## 2. Per-module dtype

HFTrainer supports two local dtype controls.

### 2.1 Artifact-loader dtype

A local component loader may document `torch_dtype` or `dtype` in its
`from_pretrained` contract. For example, the repository-owned Wan encoder
accepts:

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

The class name is resolved through `MODEL_COMPONENTS` to code under
`hftrainer.models.wan`; `from_pretrained` describes an on-disk artifact, not an
external implementation.

### 2.2 Bundle post-load cast

Use `module_dtype` for a uniform `nn.Module.to(dtype=...)` cast after the local
component has been constructed:

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

- `'fp32'`, `'float32'`, `'torch.float32'`;
- `'fp16'`, `'float16'`, `'torch.float16'`;
- `'bf16'`, `'bfloat16'`, `'torch.bfloat16'`;
- a real `torch.dtype`.

For a strict policy such as `vae=fp32` and `transformer=bf16`, configure each
module and use `accelerator.mixed_precision='no'`. Global AMP may otherwise
autocast eligible operations even when parameter storage dtypes differ.

## 3. Gradient checkpointing

Any bundle sub-module may request activation checkpointing:

```python
transformer=dict(
    type='WanTransformer3DModel',
    from_pretrained=dict(
        pretrained_model_name_or_path=CKPT_PATH + '/transformer',
    ),
    gradient_checkpointing=True,
)
```

`ModelBundle` calls `gradient_checkpointing_enable(...)` or
`enable_gradient_checkpointing(...)` on the local module. A dict may be used
only when that local hook documents keyword arguments. The bundle raises an
explicit configuration error when neither hook exists or documented arguments
are rejected.

## 4. Other supported controls

- `trainable=False` freezes a module and avoids optimizer state for it.
- `trainable='lora'` injects HFTrainer's local low-rank layers and trains only
  adapters; see [LoRA](lora.md).
- `checkpoint_format='lora'` saves adapter-only checkpoints.
- `save_ckpt=False` skips a frozen module during selective save/load. This
  reduces checkpoint I/O and disk use, not runtime GPU memory.
- Optimizers resolve only from `torch.optim`. Named HFTrainer schedules use
  `hftrainer.optim.schedulers`; explicit PyTorch scheduler classes may resolve
  from `torch.optim.lr_scheduler`.

QLoRA is not exposed because HFTrainer does not yet own a validated local
4-bit linear kernel.

## 5. Model-specific options

Only pass loader options documented by the selected repository-local model
implementation. HFTrainer deliberately does not provide a generic escape hatch
that forwards arbitrary arguments to another model framework. Unsupported
options fail at the local constructor/loader boundary.

## 6. Not yet standardized

The following are not currently one cross-model config contract:

- memory-efficient attention backends;
- attention slicing and VAE tiling helpers;
- 8-bit optimizer presets;
- module-level autocast-disable or force-fp32 islands;
- packaged ZeRO/FSDP offload presets beyond Accelerate configuration.

The stable cross-model controls are:

- `accelerator.mixed_precision`;
- `accelerator.gradient_accumulation_steps`;
- locally documented `from_pretrained.torch_dtype` / `dtype`;
- `module_dtype`;
- `gradient_checkpointing`;
- `trainable`, `save_ckpt`, and `checkpoint_format`.
