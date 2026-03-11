# Checkpointing

HF-Trainer uses one checkpoint concept with two load scopes:

- `load_scope='model'`: load selective model weights only
- `load_scope='full'`: load full accelerator state for resume

## Files

`model.pt`

- nested selective state dict for modules marked `save_ckpt=True`
- includes `__hftrainer_meta__` with per-module checkpoint metadata
- LoRA modules default to adapter-only weights when `checkpoint_format='lora'`

Accelerate state files:

- `model.safetensors`
- `optimizer.bin`
- `scheduler.bin`
- `random_states_0.pkl`

`meta.pt`

- `global_step`
- `current_epoch`

## Resume Rules

- iter-based checkpoints use completed optimizer steps
- epoch-based checkpoints use completed epochs
- `auto_resume=True` picks the latest checkpoint in `work_dir`

This avoids the common off-by-one bug where resume repeats the last completed step.

## LoRA Formats

- `checkpoint_format='lora'`: adapter-only checkpoint
- `checkpoint_format='full'`: full wrapped module state dict

When a module is declared with `trainable='lora'`, HF-Trainer defaults to `checkpoint_format='lora'`.
