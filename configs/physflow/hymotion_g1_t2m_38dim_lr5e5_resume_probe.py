# Probe a modest LR increase while preserving the full AdamW state.
#
# This keeps the original work_dir so auto_resume loads the latest official
# checkpoint, then ``resume_lr_override`` updates optimizer param groups after
# Accelerator restores the optimizer state.  The checkpoint interval is set very
# high so the probe can be stopped without writing experimental checkpoints.

_base_ = 'hymotion_g1_t2m_38dim_long.py'

resume_lr_override = 5e-5

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000000,
        max_keep_ckpts=10,
        save_last=True,
    ),
)
