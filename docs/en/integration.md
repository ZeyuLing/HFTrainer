# Integrating Models

This page answers two practical questions:

1. If `diffusers` or `transformers` already implements the model, what should you change to train it with HF-Trainer?
2. If the model is not in `diffusers` / `transformers`, or is fully custom, what should you implement yourself?

## The Contract

`ModelBundle` now exposes two public construction APIs:

- `ModelBundle.from_config(cfg, **kwargs)`: generic parent-class method for config-driven construction
- `ModelBundle.from_pretrained(pretrained_model_name_or_path, **kwargs)`: generic parent-class method for HuggingFace-style loading

The split is:

- the parent class owns the public API shape
- ordinary HF-native bundles should prefer `HF_PRETRAINED_SPEC` and `HF_SAVE_PRETRAINED_SPEC`
- override `_bundle_config_from_pretrained(...)` or `save_pretrained(...)` only when the artifact layout is unusual enough that the declarative specs are not enough

## Path 1: Start From An Existing HuggingFace Model

Use this path when the core model class already exists in `transformers` or `diffusers`.

### What You Keep

- the official model class such as `AutoModelForCausalLM`, `UNet2DConditionModel`, `WanTransformer3DModel`
- the official tokenizer / processor / scheduler classes
- the official `from_pretrained(...)` semantics for individual components

### What You Add

1. A `ModelBundle` subclass that groups the task components.
2. Task atomic forward methods such as `encode_text`, `predict_noise`, `generate`, or `classify`.
3. A task trainer that defines losses and optimization order.
4. Usually just a declarative export spec, and only a custom `save_pretrained(...)` override when the export format is unusual.

Memory and precision control are also added at the bundle-config layer, not by rewriting the official model class. Common overrides include `trainable`, `save_ckpt`, `checkpoint_format`, `lora_cfg`, `gradient_checkpointing`, `from_pretrained.torch_dtype`, and `module_dtype`.

### Minimal Pattern

```python
@MODEL_BUNDLES.register_module()
class MyLMBundle(ModelBundle):
    HF_PRETRAINED_SPEC = {
        'components': {
            'model': {
                'default_type': 'AutoModelForCausalLM',
                'pretrained_kwargs_arg': 'model_kwargs',
                'overrides_arg': 'model_overrides',
            },
        },
        'init_args': {
            'tokenizer_path': {
                'default': ModelBundle._PRETRAINED_PATH_SENTINEL,
            },
        },
    }
    HF_SAVE_PRETRAINED_SPEC = {
        'kind': 'module',
        'module': 'model',
        'extra_artifacts': ['tokenizer'],
    }

    def __init__(self, model, tokenizer_path=None):
        super().__init__()
        self._build_modules({'model': model})
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
```

### How To Make It Trainable

Use `config_overrides` or nested module overrides to add training behavior on top of the official load path:

```python
bundle = CausalLMBundle.from_pretrained(
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    config_overrides=dict(
        model=dict(
            trainable='lora',
            checkpoint_format='lora',
            lora_cfg=dict(
                task_type='CAUSAL_LM',
                target_modules='all-linear',
                r=16,
                lora_alpha=32,
            ),
        ),
    ),
    max_length=1024,
)
```

For diffusion-style tasks the pattern is the same, but the bundle expands one pretrained directory into multiple components:

```python
bundle = SD15Bundle.from_pretrained(
    '/path/to/stable-diffusion-v1-5',
    unet_overrides=dict(trainable=True, save_ckpt=True),
    text_encoder_overrides=dict(trainable=False, save_ckpt=False),
)
```

See [Memory and Precision](memory.md) for the full config contract and caveats around global AMP vs per-module dtype.

### Export Back To Official Inference APIs

HF-native bundles should usually declare `HF_SAVE_PRETRAINED_SPEC`. Only special export formats need a hand-written `save_pretrained(...)`.

Current implementations:

- `CausalLMBundle.save_pretrained(...)`
- `ViTBundle.save_pretrained(...)`
- `SD15Bundle.save_pretrained(...)`
- `WanBundle.save_pretrained(...)`

Examples:

```python
bundle.save_pretrained('exports/tinyllama', merge_lora=True)
model = AutoModelForCausalLM.from_pretrained('exports/tinyllama')
```

```python
bundle.save_pretrained('exports/sd15')
pipe = StableDiffusionPipeline.from_pretrained('exports/sd15')
```

If you save a LoRA bundle with `merge_lora=False`, the result stays adapter-style and should be loaded with PEFT-native APIs instead of plain `AutoModel...`.

## Path 2: Start From A Custom Or Self-Developed Model

Use this path when HuggingFace does not already provide the core model class.

### What You Implement

1. Your own `nn.Module` or family of modules.
2. Registration in `HF_MODELS` or direct use of the class in config.
3. A `ModelBundle` subclass that calls `_build_modules(...)`.
4. Task atomic forward methods.
5. A task trainer.

### Minimal Pattern

```python
class MyBackbone(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        ...

@MODEL_BUNDLES.register_module()
class MyCustomBundle(ModelBundle):
    def __init__(self, backbone, head):
        super().__init__()
        self._build_modules({
            'backbone': backbone,
            'head': head,
        })

    def forward_features(self, x):
        x = self.backbone(x)
        return self.head(x)
```

Construct it with the generic parent-class API:

```python
bundle = MyCustomBundle.from_config(
    dict(
        backbone=dict(type=MyBackbone, hidden_size=1024),
        head=dict(type=MyHead, num_classes=1000),
    )
)
```

### When To Implement `from_pretrained(...)` Yourself

You only need a bundle-specific pretrained path when you want one of these:

- users can type `MyCustomBundle.from_pretrained(...)`
- your exported artifact has a stable on-disk format
- you want to bridge your custom task to a third-party inference API

In that case:

1. try `HF_PRETRAINED_SPEC` / `HF_SAVE_PRETRAINED_SPEC` first
2. only if the spec cannot express the mapping, implement `_bundle_config_from_pretrained(...)`
3. implement `save_pretrained(...)` only when export also needs task-specific logic
4. document the exported artifact format

If you do not need those guarantees, `from_config(...)` plus checkpoint save/load is enough.

## Design Rule

Keep the public API simple:

- use `from_pretrained(...)` when there is already a HuggingFace-native model artifact
- use `from_config(...)` when the model is custom or the task is not naturally represented by one official artifact
- do not wrap official model classes just to rename their semantics
