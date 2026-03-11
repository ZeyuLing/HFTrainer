# API Reference

This page documents the public, user-facing API surface of HF-Trainer. It focuses on the classes, methods, and CLI entry points that users are expected to instantiate, subclass, or call directly.

It does not try to list every internal helper function in the repository. Internal task-specific atomic methods are still best understood from the source of each task bundle.

## Coverage

This reference covers:

- framework entry points: `AccelerateRunner`, `ModelBundle`, `BaseTrainer`, `BasePipeline`
- validation/runtime extension points: hooks, evaluators, visualizers
- checkpoint utilities: `find_latest_checkpoint`, `load_checkpoint`, `save_checkpoint`
- command-line entry points: `tools/train.py`, `tools/infer.py`

For runnable task stacks, see [Task Matrix](tasks.md).

## `AccelerateRunner`

Import:

```python
from hftrainer import AccelerateRunner
```

Typical usage:

```python
runner = AccelerateRunner.from_cfg(cfg)
runner.train()
```

### Main API

| Method | What it does | Inputs |
| --- | --- | --- |
| `from_cfg(cfg)` | build the full runtime from config | `cfg`: `dict` or `mmengine.Config` |
| `train()` | run the training loop | none |
| `val()` | run one validation pass | none |
| `save_checkpoint()` | save a checkpoint under `work_dir` | none |

### `from_cfg(cfg)` expects

| Config key | Required | Purpose |
| --- | --- | --- |
| `model` | yes | bundle config |
| `trainer` | yes | trainer config |
| `train_dataloader` | yes for training | training dataset / dataloader config |
| `optimizer` | yes for training | optimizer config, single or named dict |
| `train_cfg` | recommended | iter/epoch loop settings |
| `lr_scheduler` | optional | scheduler config |
| `default_hooks` | optional | hook config |
| `val_dataloader` | optional | validation dataloader config |
| `val_evaluator` | optional | evaluator config |
| `val_visualizer` | optional | visualizer config |
| `accelerator` | optional | accelerate runtime config |
| `work_dir` | optional | experiment root |
| `auto_resume` | optional | resume latest checkpoint in `work_dir` |
| `load_from` | optional | explicit checkpoint load config |

### Important runner attributes

| Attribute | Meaning |
| --- | --- |
| `bundle` | instantiated `ModelBundle` |
| `trainer` | instantiated trainer |
| `hooks` | built and priority-sorted hooks |
| `evaluators` | validation metric components |
| `visualizers` | validation visualization components |
| `optimizers` | named optimizer dict |
| `lr_schedulers` | named scheduler dict |
| `global_step` | completed optimizer steps |
| `current_epoch` | completed epochs |
| `work_dir` | checkpoint root |
| `run_dir` | timestamped log / config / tensorboard dir |

## `ModelBundle`

Import:

```python
from hftrainer.models import ModelBundle
```

Purpose:

- hold all task sub-modules
- instantiate them from config
- control per-module trainability and checkpoint behavior
- expose shared atomic forward functions for both trainer and pipeline

### Sub-module config keys

Each bundle sub-module can declare:

| Key | Type | Meaning |
| --- | --- | --- |
| `type` | `str` | registry key / class name |
| `from_pretrained` | `dict` | HuggingFace-style pretrained load arguments |
| `from_config` | `dict` | config-based construction arguments |
| `from_single_file` | `dict` | single-file load arguments |
| `trainable` | `bool` or `'lora'` | train full module, freeze it, or inject LoRA |
| `save_ckpt` | `bool` | include module in selective checkpoint save/load |
| `checkpoint_format` | `'full'` or `'lora'` | save full weights or adapter-only weights |
| `lora_cfg` | `dict` | PEFT LoRA configuration when `trainable='lora'` |
| `gradient_checkpointing` | `bool` or `dict` | enable activation checkpointing on modules that expose a supported HF hook |
| `module_dtype` | dtype string or `torch.dtype` | post-load cast applied to the constructed sub-module |

### Main API

| Method | What it does | Inputs |
| --- | --- | --- |
| `from_config(cfg=None, **kwargs)` | generic parent-class bundle construction | config dict or `mmengine.ConfigDict` |
| `from_pretrained(path, config_overrides=None, **kwargs)` | generic HuggingFace-style bundle construction | pretrained path plus task-specific kwargs |
| `_build_modules(modules_cfg)` | instantiate and configure sub-modules | `modules_cfg: dict` |
| `trainable_parameters()` | return all trainable parameters | none |
| `trainable_named_parameters()` | yield trainable `(name, param)` pairs | none |
| `get_module_parameters(*names)` | collect parameters from named sub-modules | module names |
| `get_module_build_cfg(name)` | return the normalized build config of one sub-module | sub-module name |
| `get_module_pretrained_path(name)` | return the recorded `from_pretrained` path for one sub-module | sub-module name |
| `state_dict_to_save()` | build selective save dict | none |
| `load_state_dict_selective(state_dict, strict=False)` | load only modules present in checkpoint | nested or flat state dict |
| `checkpoint_metadata()` | describe saved modules and formats | none |
| `merge_lora_weights(module_names=None)` | merge adapter weights into base modules | optional module-name list |
| `save_pretrained(save_directory, **kwargs)` | export task-specific inference artifact | export directory |

### Construction semantics

- `from_config(...)` is fully generic and should be the default entry for custom/self-developed models.
- `from_pretrained(...)` is a generic parent-class API, but each HF-native bundle still defines how one pretrained artifact maps to sub-modules through `_bundle_config_from_pretrained(...)`.
- `save_pretrained(...)` is intentionally task-specific. Implement it only when the bundle can export an artifact that official inference APIs can read.
- `from_pretrained.torch_dtype` / `dtype` are passed through to the underlying HF loader, while `module_dtype` is an HF-Trainer post-load cast.
- See [Memory and Precision](memory.md) for global AMP, per-module dtype, and gradient-checkpointing guidance.

### `load_state_dict_selective(...)`

Accepted inputs:

- nested save format: `{module_name: module_state_dict}`
- flat checkpoint dicts that can be split by module prefix
- LoRA adapter-only checkpoints written with `checkpoint_format='lora'`

### `merge_lora_weights(...)`

Use this before export or inference when you want LoRA adapters merged into base weights:

```python
bundle.merge_lora_weights()
```

If `module_names=None`, all LoRA sub-modules in the bundle are merged.

## `BaseTrainer`

Import:

```python
from hftrainer import BaseTrainer
```

Subclass this for task training logic.

### Main API

| Method | What it does | Inputs |
| --- | --- | --- |
| `train_step(batch)` | one training step | dataloader batch dict |
| `val_step(batch)` | one validation step | dataloader batch dict |
| `set_optimizers(optimizers, lr_schedulers=None)` | inject named optimizers/schedulers | dicts from runner |
| `get_optimizer(name)` | fetch a named optimizer | optimizer name |
| `get_lr_scheduler(name)` | fetch a named scheduler | scheduler name |
| `get_global_step()` | completed training steps before current iter | none |
| `get_current_step()` | 1-based current step index | none |
| `get_discriminator_factor(...)` | delayed-start / warmup factor helper | weight and schedule args |
| `should_update_discriminator(...)` | cadence helper for discriminator steps | start and interval args |

### Multi-optimizer protocol

Set:

```python
trainer_controls_optimization = True
```

When enabled:

- the runner injects optimizers and schedulers
- the runner skips its default backward / step / zero_grad path
- `train_step()` must call `self.accelerator.backward(...)` and step optimizers directly
- `train_step()` should return `{'loss': None, ...}` plus logging values

## `BasePipeline`

Import:

```python
from hftrainer.pipelines import BasePipeline
```

Pipelines assemble inference-time control flow and always consume a bundle.

### Main API

| Method | What it does | Inputs |
| --- | --- | --- |
| `__init__(bundle, **kwargs)` | construct pipeline from bundle | task bundle |
| `from_config(bundle_cls, bundle_cfg, **kwargs)` | build bundle from config and wrap it in a pipeline | bundle class, bundle config |
| `from_pretrained(bundle_cls, path, bundle_kwargs=None, **kwargs)` | build bundle from pretrained artifact and wrap it in a pipeline | bundle class, pretrained path |
| `from_checkpoint(bundle_cls, bundle_cfg, ckpt_path, **kwargs)` | construct bundle and load checkpoint | bundle class, bundle config, checkpoint path |
| `__call__(...)` | run inference | task-specific |

## Hooks

Built-in hooks are documented in detail in [Hook System](design/hooks.md). Public built-ins are:

| Hook | Purpose | Key arguments |
| --- | --- | --- |
| `LoggerHook` | print and log scalar metrics | `interval`, `by_epoch` |
| `CheckpointHook` | periodic and final checkpoint save | `interval`, `max_keep_ckpts`, `save_last`, `by_epoch` |
| `EMAHook` | maintain EMA copy of trainable modules | `decay`, `update_interval` |
| `LRSchedulerHook` | compatibility placeholder | `by_epoch` |

## Evaluators and Visualizers

### `BaseEvaluator`

Import:

```python
from hftrainer.evaluation import BaseEvaluator
```

| Method | What it does |
| --- | --- |
| `reset()` | clear accumulated validation results |
| `process(output)` | consume one `val_step()` output dict |
| `compute()` | return final metric dict |
| `compute_from_outputs(outputs)` | convenience helper for full-output lists |

### `BaseVisualizer`

Import:

```python
from hftrainer.visualization import BaseVisualizer
```

| Method | What it does |
| --- | --- |
| `visualize(output, step)` | render one validation output dict |
| `should_visualize(step)` | interval-based gating helper |

## Checkpoint Utilities

Import:

```python
from hftrainer.utils.checkpoint_utils import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
```

| Function | What it does | Inputs |
| --- | --- | --- |
| `find_latest_checkpoint(work_dir)` | locate newest checkpoint dir | experiment root |
| `load_checkpoint(path, map_location='cpu')` | load a file or checkpoint directory | file path or checkpoint dir |
| `save_checkpoint(state_dict, path, use_safetensors=True)` | write a state dict to disk | state dict and output path |

## CLI Reference

### `tools/train.py`

Usage:

```bash
python3 tools/train.py CONFIG.py [OPTIONS]
```

| Argument | Meaning |
| --- | --- |
| `config` | config file path |
| `--work-dir` | override `cfg.work_dir` |
| `--auto-resume` | resume latest checkpoint in `work_dir` |
| `--load-from` | explicit checkpoint path |
| `--load-scope` | `model` or `full` |
| `--cfg-options` | override config values from CLI |

### `tools/infer.py`

Usage:

```bash
python3 tools/infer.py --config CONFIG.py --checkpoint CKPT_DIR [OPTIONS]
```

| Argument | Meaning |
| --- | --- |
| `--config` | config file path |
| `--checkpoint` | checkpoint directory or checkpoint file |
| `--prompt` | text prompt for generation tasks |
| `--input` | input file for classification-style tasks |
| `--output` | output image / video path |
| `--num-steps` | diffusion denoising steps |
| `--num-samples` | unconditional sample count |
| `--num-frames` | video frame count |
| `--max-new-tokens` | LLM generation length |
| `--height`, `--width` | output spatial size |
| `--merge-lora` | merge LoRA adapters into base weights before infer |
| `--device` | `cuda` or `cpu` |

## Task Class Map

| Task | Bundle | Trainer | Pipeline |
| --- | --- | --- | --- |
| Classification | `ViTBundle` | `ClassificationTrainer` | `ClassificationPipeline` |
| Text-to-image | `SD15Bundle` | `SD15Trainer` | `SD15Pipeline` |
| Causal LM | `CausalLMBundle` | `CausalLMTrainer` | `CausalLMPipeline` |
| Text-to-video | `WanBundle` | `WanTrainer` | `WanPipeline` |
| GAN | `StyleGAN2Bundle` | `GANTrainer` | `StyleGAN2Pipeline` |
| DMD | `DMDBundle` | `DMDTrainer` | `DMDPipeline` |

For runnable configs and task-specific notes, see [Task Matrix](tasks.md).
