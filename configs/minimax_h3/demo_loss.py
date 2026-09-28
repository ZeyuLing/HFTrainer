"""20-step real-data LoRA trace; not a convergence benchmark."""
_base_ = ["validate_h3_local.py"]
train_cfg = dict(max_iters=20, val_interval=5)
default_hooks = dict(checkpoint=dict(interval=20, max_keep_ckpts=1))
