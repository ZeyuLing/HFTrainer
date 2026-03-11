# ModelBundle

`ModelBundle` is the shared task core used by both training and inference.

## Responsibilities

- build sub-modules from config
- apply `trainable` / `save_ckpt` flags per sub-module
- expose task-specific atomic forward functions
- provide selective checkpoint save/load
- expose one generic `from_config(...)` entry point for all bundles
- expose one generic `from_pretrained(...)` entry point for HF-native bundles

## Why It Exists

Without a shared bundle, the same task logic tends to be duplicated in both trainers and inference pipelines. HF-Trainer keeps shared logic in the bundle and lets each trainer or pipeline only assemble the control flow around it.

## Current Examples

- `ViTBundle`
- `SD15Bundle`
- `CausalLMBundle`
- `WanBundle`

## Construction Rules

`ModelBundle` now has one parent-class construction contract:

- `from_config(...)` is generic and works for all bundles
- `from_pretrained(...)` is also defined on the parent class, but HF-native bundles supply `_bundle_config_from_pretrained(...)`
- `save_pretrained(...)` is task-specific and should only be implemented when the bundle can export an artifact official inference APIs understand

That means users only need to learn one public shape, while bundle subclasses only fill in task-specific mapping logic.

## Two Integration Paths

### Path A: HuggingFace-native models

If `transformers` or `diffusers` already has the model class:

- keep the official class inside the bundle
- use `Bundle.from_pretrained(...)` as the public entry
- add training behavior through config overrides such as `trainable`, `save_ckpt`, `checkpoint_format`, and `lora_cfg`
- implement `save_pretrained(...)` if you want the training result to round-trip through official inference APIs

### Path B: Custom or self-developed models

If the model does not exist in HuggingFace libraries:

- implement your own `nn.Module`
- instantiate it through `Bundle.from_config(...)`
- only add bundle-specific `from_pretrained(...)` / `save_pretrained(...)` if you need a stable exported artifact

See [Integration Guide](../integration.md) for concrete examples.

## Per-Module Control

Each sub-module can declare:

- `trainable=True/False/'lora'`
- `save_ckpt=True/False`
- `checkpoint_format='full'|'lora'`
- `gradient_checkpointing=True` or a dict of kwargs
- `module_dtype='fp32'|'fp16'|'bf16'` (or a `torch.dtype`)
- `from_pretrained` / `from_config` / `from_single_file`

That lets checkpoints skip frozen modules and lets optimizers only see trainable parameters.

Use `from_pretrained.torch_dtype` when the underlying HF loader already supports the target dtype, and `module_dtype` when you want HF-Trainer to cast the constructed module after load. See [Memory and Precision](../memory.md) for examples and caveats.

## LoRA

Recommended config pattern:

```python
model = dict(
    type='CausalLMBundle',
    model=dict(
        type='AutoModelForCausalLM',
        from_pretrained=dict(pretrained_model_name_or_path=CKPT_PATH),
        trainable='lora',
        checkpoint_format='lora',
        lora_cfg=dict(
            task_type='CAUSAL_LM',
            target_modules='all-linear',
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
        ),
    ),
)
```

LoRA modules default to `checkpoint_format='lora'`, which means adapter-only save/load. Use `checkpoint_format='full'` if you explicitly want the full wrapped module state dict instead.
