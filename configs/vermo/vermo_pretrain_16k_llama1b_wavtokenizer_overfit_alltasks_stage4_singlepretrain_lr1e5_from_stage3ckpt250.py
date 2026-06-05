"""Single-pretrain-target all-task overfit validation.

MotionPretrainTask is unconditional: it has an empty template and no input
modalities. Multiple unique pretrain targets therefore share the same prompt
and cannot all be exact-match reconstructed. This config keeps 190 overfit
slots while using one repeated pretrain source plus 18 conditioned tasks x 10.
"""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage3_all190_lr5e6_from_focusckpt150.py'

work_dir = (
    'work_dirs/'
    'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage4_singlepretrain_lr1e5_from_stage3ckpt250'
)

auto_resume = False
load_from = dict(
    _delete_=True,
    path=(
        'work_dirs/'
        'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage3_all190_lr5e6_from_focusckpt150/'
        'checkpoint-iter_250'
    ),
    load_scope='model',
)

train_dataloader = dict(
    dataset=dict(
        anno_file='data/annotation/vermo_overfit_alltasks_190_singlepretrain_20260603.json',
    ),
)

optimizer = dict(
    type='AdamW',
    lr=1e-5,
    betas=[0.9, 0.99],
    weight_decay=0.0,
)

train_cfg = dict(
    by_epoch=False,
    max_iters=300,
    val_interval=10000,
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=1),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=50,
        max_keep_ckpts=8,
        save_last=True,
    ),
)
