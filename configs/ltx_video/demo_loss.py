"""20-step real-data LoRA trace; no expensive generated validation clips."""
_base_ = ["validate_ltx_2_5_local.py"]
trainer = dict(native_config=dict(
    optimization=dict(steps=20),
    validation=dict(samples=[], interval=None),
    checkpoints=dict(interval=20, keep_last_n=1),
))
