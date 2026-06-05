"""Single-GPU accumulated stage-2 VerMo overfit validation.

This variant smooths the output-only stage-2 updates by accumulating several
single-sample tasks before each optimizer step.
"""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage2_outputonly_lr2e5.py'

work_dir = (
    'work_dirs/'
    'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage2_outputonly_lr1e5_from1250_1gpu_accum8_skipguard'
)

accelerator = dict(
    mixed_precision='fp16',
    gradient_accumulation_steps=8,
)

optimizer = dict(
    type='AdamW',
    lr=1e-5,
    betas=[0.9, 0.99],
    weight_decay=0.0,
)

train_cfg = dict(
    by_epoch=False,
    max_iters=2400,
    val_interval=10000,
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=1),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=100,
        max_keep_ckpts=8,
        save_last=True,
    ),
)
