# Architecture

## Overview

```
Config (.py)
  |
  v
AccelerateRunner.from_cfg(cfg)
  |
  +-- ModelBundle          <-- holds all sub-modules + atomic forward functions
  |     |                      (encode_text, predict_noise, decode_latent, ...)
  |     |
  |     +-- Trainer        <-- training logic only: assemble loss, call bundle methods
  |     +-- Pipeline       <-- inference logic only: denoising loop, call bundle methods
  |
  +-- Accelerator          <-- DDP / FSDP / DeepSpeed / mixed precision
  +-- DataLoader(s)
  +-- Optimizer(s)         <-- single or named dict (GAN support)
  +-- LR Scheduler(s)
  +-- Hooks                <-- checkpoint, logger, EMA, etc.
  +-- Evaluator(s)         <-- consume val_step output dicts
  +-- Visualizer(s)        <-- log images/text to tensorboard
```

## AccelerateRunner

The central orchestrator. `AccelerateRunner.from_cfg(cfg)` builds all components from a single config:

1. Create `work_dir` + timestamped `run_dir`
2. Build `ModelBundle` (via `HF_MODELS` registry + `_build_modules`: instantiate, freeze, apply LoRA)
3. Build `Trainer` (via `TRAINERS` registry, inject the bundle)
4. Build `DataLoader` (via `DATASETS` registry)
5. Build `Optimizer(s)` (single or named dict, from `bundle.trainable_parameters()`)
6. Build `LR Scheduler(s)`
7. Build `Hooks` + `Evaluator(s)` + `Visualizer(s)`
8. Create `Accelerator`
9. `accelerator.prepare(trainable_modules, optimizer(s), dataloader, scheduler(s))`
   -- only trainable modules go through `prepare`; frozen modules are untouched
10. Handle `load_from` / `auto_resume`
11. Run the training loop

## ModelBundle

The shared core between Trainer and Pipeline. Holds all sub-modules and implements atomic forward functions that both Trainer and Pipeline call.

```
ModelBundle
  +-- _build_modules(cfg)           <-- instantiate sub-modules, set freeze/lora, record save_ckpt
  +-- trainable_parameters()        <-- only return trainable module params (for optimizer)
  +-- state_dict_to_save()          <-- only return save_ckpt=True module weights
  +-- load_state_dict_selective()   <-- only load modules present in state_dict
  +-- [task-specific atomic functions]
       encode_text(texts) -> Tensor
       encode_image/video(x) -> latent
       predict_noise(latent, t, cond) -> Tensor
       decode_latent(latent) -> image/video
```

## Per-Module Config Control

Each sub-module independently declares its loading method, trainability, and checkpoint behavior:

```python
model = dict(
    type='WanModelBundle',
    text_encoder=dict(..., trainable=False, save_ckpt=False),  # frozen, skip IO
    vae=dict(..., trainable=False, save_ckpt=False),
    transformer=dict(..., trainable=True, save_ckpt=True),     # only this trains & saves
    scheduler=dict(..., trainable=False, save_ckpt=False),
)
```

- `trainable=True` -- parameters participate in training (gradients computed)
- `trainable=False` -- frozen (`requires_grad_(False)`)
- `trainable='lora'` -- LoRA injected via `peft`, only LoRA params are trainable
- `save_ckpt=True` -- included in checkpoint save/load
- `save_ckpt=False` -- skipped during checkpoint I/O (saves time and disk)

## Trainer

Only responsible for training logic. Assembles the training forward graph, computes loss, and returns a loss dict. All forward functions are called through `self.bundle`.

```python
class WanTrainer(BaseTrainer):
    def train_step(self, batch) -> dict:
        text_emb = self.bundle.encode_text(batch['text'])
        latent   = self.bundle.encode_video(batch['video'])
        ...
        return {'loss': loss}
```

## Pipeline

Only responsible for inference logic. Assembles the inference forward graph (e.g., denoising loop) and post-processes output. All forward functions are called through `self.bundle`.

```python
class WanPipeline(BasePipeline):
    @torch.no_grad()
    def __call__(self, text, num_steps=50) -> Tensor:
        text_emb = self.bundle.encode_text([text])
        ...
        return self.bundle.decode_latent(latent)
```

## Registry System

```python
HF_MODELS    = Registry('hf_model')      # HuggingFace native classes (from_pretrained etc.)
MODEL_BUNDLES = Registry('model_bundle')  # ModelBundle subclasses
TRAINERS     = Registry('trainer')        # Trainer subclasses
PIPELINES    = Registry('pipeline')       # Pipeline subclasses
DATASETS     = Registry('dataset')
TRANSFORMS   = Registry('transform')
HOOKS        = Registry('hook')
EVALUATORS   = Registry('evaluator')
VISUALIZERS  = Registry('visualizer')
```

See [Design Documentation](design/index.md) for full design rationale.
