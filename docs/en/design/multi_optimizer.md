# Multi-Optimizer

## Scope

HF-Trainer supports multi-optimizer execution at the runner and trainer API level, and now includes runnable reference projects for StyleGAN2-style GAN training and DMD-style distillation.

Implemented framework pieces:

- named `optimizer` dicts
- named `lr_scheduler` dicts
- `BaseTrainer.trainer_controls_optimization = True`
- optimizer / scheduler injection through `set_optimizers()`
- discriminator schedule controls:
  - `disc_start_step`
  - `disc_warmup_steps`
  - `disc_update_interval`

Existing reference trainers:

- `StyleGAN2Trainer`
- `DMDTrainer`

## Discriminator Scheduling

`StyleGAN2Trainer` and `DMDTrainer` now support delayed discriminator activation.

Use these trainer fields:

- `disc_start_step`: number of completed training steps before adversarial training becomes active
- `disc_warmup_steps`: linear ramp from `0` to the target adversarial weight
- `disc_update_interval`: discriminator optimizer step cadence after activation

Semantics:

- the schedule is evaluated against completed global steps, so `disc_start_step=5000` means the discriminator first affects step 5001
- `disc_update_interval` is counted from the first active discriminator step, so delayed start does not shift the update phase

## Intended Usage

When a trainer needs alternating phases, the runner should not apply its default single-loss optimization path. Instead, the trainer performs:

- `self.accelerator.backward(loss)`
- `optimizer.step()`
- `optimizer.zero_grad()`
- optional scheduler stepping

and returns `{'loss': None, ...}` so the runner does not step twice.

## Current Limitation

- The GAN and DMD stacks are reference implementations aimed at framework integration and algorithm structure, not benchmark-tuned reproductions.
- DMD can consume precomputed regression pairs, but the demo config also supports online teacher target generation for convenience.

## Optimizer classes and Muon

Set `optimizer.type` to a PyTorch class name such as `AdamW`, or use the fully
qualified import path of an installed `torch.optim.Optimizer` subclass. The
optimizer's constructor receives the remaining configuration fields.

```python
optimizer = dict(
    type='transformers.optimization.Adafactor',
    lr=1e-4,
    relative_step=False,
    scale_parameter=False,
)
```

`type='Muon'` uses `torch.optim.Muon` from PyTorch 2.9+. The runner assigns
two-dimensional hidden weights to Muon and remaining weights to AdamW.
Use `exclude` to send additional parameters to AdamW by name regex and
`adamw` to override its learning rate or other constructor arguments.
`adjust_lr_fn='match_rms_adamw'` selects PyTorch's Moonshot scaling.

For example, select Muon in an LTX-Video run with:

```bash
python tools/train.py configs/ltx_video/train_ltx_video_2_5_lora.py --cfg-options \
  optimizer.type=Muon optimizer.lr=4e-4 optimizer.adamw.lr=4e-5 \
  optimizer.adjust_lr_fn=match_rms_adamw
```

Automatic grouping works with runner-controlled optimization on a single device
or with DDP. Sharded FSDP and DeepSpeed training require a backend-compatible
optimizer. Use `--load-from PATH --load-scope model` to initialize a run from
existing weights when changing the optimizer.
