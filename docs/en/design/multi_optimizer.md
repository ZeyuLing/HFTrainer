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

- `GANTrainer`
- `DMDTrainer`

## Discriminator Scheduling

`GANTrainer` and `DMDTrainer` now support delayed discriminator activation.

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
