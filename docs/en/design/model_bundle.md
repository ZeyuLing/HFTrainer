# ModelBundle

`ModelBundle` is the boundary between repository-owned model mathematics and
training/inference orchestration.

## What belongs in a bundle

- explicit local component construction;
- component freeze/train/LoRA/checkpoint policy;
- atomic operations shared by trainer and pipeline;
- strict artifact loading and saving;
- model-family invariants and shape/config validation.

## What does not belong in a bundle

- imports of another model implementation package;
- arbitrary dotted-class resolution;
- complete training loops or optimizer stepping;
- CLI parsing, output encoding, or visualization;
- a second copy of pipeline-only denoising logic.

## Local components

`_build_modules()` resolves component names only through `MODEL_COMPONENTS`.
Unknown names fail with the currently registered local names. Dotted paths are
not interpreted as import instructions.

Each module config may use:

```python
transformer=dict(
    type='MyTransformer',
    from_pretrained=dict(
        pretrained_model_name_or_path='checkpoints/my-method/transformer',
    ),
    trainable='lora',       # True, False, or 'lora'
    lora_cfg=dict(rank=16, alpha=16, target_modules=['to_q', 'to_v']),
    save_ckpt=True,
    checkpoint_format='lora',
    module_dtype='bf16',
    gradient_checkpointing=True,
)
```

`from_pretrained` here means “load a supported artifact into the local class.”
It does not select an external class.

## `PRETRAINED_SPEC`

Simple bundles can declare how one artifact root maps to local components:

```python
class MyBundle(ModelBundle):
    PRETRAINED_SPEC = {
        'components': {
            'model': {
                'default_type': 'MyLocalModel',
                'subfolder': 'model',
                'overrides_arg': 'model_overrides',
            },
        },
        'init_args': {
            'tokenizer_path': {
                'default': ModelBundle._PRETRAINED_PATH_SENTINEL,
            },
        },
    }
```

The resolved class still comes from `MODEL_COMPONENTS`. Complex families may
override `_bundle_config_from_pretrained()` when one artifact needs explicit
role checks or conversion.

## Export

Every concrete bundle implements `save_pretrained()`. There is deliberately no
generic “import a pipeline class and ask it to save” mechanism. The concrete
implementation must define:

- configuration schema;
- tensor files and shard indexes;
- tokenizer/processor assets;
- shared-weight aliases;
- manifest, hashes, and version;
- exact reload validation.

This makes the training artifact independent of whichever model package is
installed in the inference environment.

## Atomic operations

A diffusion bundle might expose:

```python
encode_text(prompts)
encode_image(images)
add_noise(latents, noise, timesteps)
predict_noise(noisy_latents, timesteps, conditioning)
decode_latent(latents)
```

The trainer combines these into a loss. The pipeline combines the same
operations into sampling. Neither should reach through the bundle and recreate
private component behavior.

## Checkpoint scopes

HFTrainer checkpoints can save selected bundle components and direct
bundle-level parameters. A component has one explicit format:

- `full`: component state dict;
- `lora`: local adapter-only state dict.

Frozen components can be omitted when they are reproducibly loaded from their
base artifact. Bundle artifact export is separate and should produce a complete
inference artifact according to the implementation's schema.

## Invariants enforced by tests

- model-layer source contains no dynamic import escape hatch;
- registered built-in components are defined below `hftrainer.models`;
- frozen modules remain in eval mode;
- LoRA state loads without silent missing/unexpected keys;
- artifact round-trip preserves outputs;
- trainer and pipeline configs resolve independently.
