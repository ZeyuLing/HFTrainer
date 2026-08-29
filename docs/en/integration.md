# Integrating a model

HFTrainer integrations are source integrations, not runtime wrappers around a
second model framework. A model is integrated only when the executable core is
owned by this repository and follows the same layer boundaries as the existing
implementations.

## Non-negotiable boundary

Model code shipped by HFTrainer must not import, dynamically resolve, or
delegate execution to another model implementation package. This includes
tokenizers, adapter layers, schedulers, model classes, and pipeline objects
that determine numerical behavior.

General infrastructure dependencies remain allowed: PyTorch, Accelerate,
MMEngine, safetensors, NumPy, Pillow, torchvision, and media/scientific helper
libraries. The distinction is ownership of model execution, not a requirement
to rewrite tensor or operating-system primitives.

The CI boundary test blocks the forbidden model packages in a fresh process.
Moving an import into `bundle.py`, hiding it behind `importlib`, or loading a
dotted class path does not satisfy the contract.

## Canonical vertical slice

Use one stable implementation identifier across all method-specific layers:

```text
hftrainer/models/my_method/
  __init__.py
  bundle.py
  checkpoint.py       # when the artifact schema is non-trivial
  network/
    __init__.py
    configuration.py
    modeling.py
    tokenization.py   # when model-specific

hftrainer/trainers/my_method/
  trainer.py

hftrainer/pipelines/my_method/
  pipeline.py

configs/my_method/
  train.py
  infer.py
```

Use `hftrainer/tasks/<task_contract>` only when the trainer or pipeline is
genuinely reusable across multiple implementations. ViT uses the reusable
`image_classification` contract; a method-specific diffusion algorithm should
not be placed under a generic `text_to_image` model directory.

## Responsibilities

### `network/`

Owns model mathematics and model-specific primitives:

- layers and forward functions;
- configuration objects;
- tokenizer/processor logic required by the model;
- sampling schedulers whose behavior is part of the method;
- checkpoint-key-compatible module names.

It must not know about an experiment runner, dataloader, CLI, or visualization.

### `bundle.py`

Owns the implementation boundary:

- constructs only explicit classes from the local network package;
- validates component combinations;
- exposes atomic operations shared by training and inference;
- controls trainable/frozen/LoRA components;
- loads and saves the implementation's strict artifact schema.

`ModelBundle.PRETRAINED_SPEC` may describe how one supported artifact directory
maps to local components. It never authorizes arbitrary class imports. Export
is always implemented by the concrete bundle because the bundle must define
and validate its own schema.

### Trainer

Owns losses, update order, optimizer grouping, and training-only validation.
It calls bundle operations instead of recreating model forwards.

### Pipeline

Owns the inference graph and public inputs/outputs. The generic CLI selects a
pipeline from `cfg.pipeline.type` and an I/O adapter from `cfg.inference.task`;
it never dispatches from a trainer name.

## Component registration

Register executable components in `MODEL_COMPONENTS`:

```python
from hftrainer.registry import MODEL_COMPONENTS


@MODEL_COMPONENTS.register_module()
class MyTransformer(torch.nn.Module):
    ...
```

The class must be defined under `hftrainer.models.<implementation>`. Configs
refer to the registered local name:

```python
model = dict(
    type='MyMethodBundle',
    transformer=dict(
        type='MyTransformer',
        from_pretrained=dict(
            pretrained_model_name_or_path='checkpoints/my-method/transformer',
        ),
        trainable=True,
        save_ckpt=True,
    ),
)
```

Unknown names and dotted paths are rejected. Do not add a fallback that imports
whatever package happens to be installed.

## Importing an existing public implementation

Public source can be used as a reference or, when its license permits,
incorporated as a pinned modified snapshot. In both cases:

1. Record the repository, immutable revision, and license.
2. Preserve required copyright and attribution notices.
3. Mark modified files when the license requires it.
4. Relocate code into HFTrainer's model/trainer/pipeline boundaries.
5. Rewrite internal imports to local namespaces.
6. Remove runtime construction through external model packages.
7. Add parity tests against the reference in an isolated development
   environment; do not install the reference as a product dependency.

Never describe incorporated or reference-derived code as original HFTrainer
work. LTX-2.5 is the concrete example: its pinned source, modification record,
and separate license are included with the package.

## Artifact contract

A local artifact should normally contain:

```text
artifact/
  config.json or bundle_config.json
  model.safetensors (or an indexed shard set)
  tokenizer/processor assets when required
  manifest.json
```

The loader should validate, in proportion to the artifact:

- schema/format version;
- component class and configuration;
- expected state-dict keys and tensor shapes;
- shard or file hashes when a manifest is present;
- tied/shared parameter aliases;
- weight coverage, rejecting dangerously low coverage by default.

Do not treat `strict=False` as compatibility. If conversion is required,
provide an explicit conversion tool and record the source format.

## LoRA

Use `hftrainer.models.lora.apply_lora`. It injects local `LoRALinear` modules,
saves adapter-only state, and supports deterministic merge for inference.
QLoRA is intentionally unavailable until HFTrainer owns and validates a local
4-bit linear path.

## Required tests

Every new integration needs:

1. tiny forward with realistic tensor ranks;
2. training loss and backward;
3. tiny inference/sampling path;
4. save/load round-trip with output equivalence;
5. malformed/missing/tampered artifact rejection;
6. config import and registry resolution;
7. fresh-process import with external model packages blocked;
8. reference key/shape and numerical parity where a reference exists.

Large gated models may use contract/tiny tests when production weights cannot
be allocated locally, but the documentation must state exactly what was and
was not run.

## Review checklist

- [ ] one implementation identifier across model/trainer/pipeline/config;
- [ ] model math is under `network/`;
- [ ] no executable external model import or dynamic escape hatch;
- [ ] bundle imports explicit local components;
- [ ] trainer and pipeline share bundle operations;
- [ ] artifact coverage and mismatches are visible;
- [ ] provenance and licenses are present;
- [ ] tiny train/infer/round-trip tests pass;
- [ ] all leaf configs resolve local component classes;
- [ ] wheel installs and imports without forbidden packages.
