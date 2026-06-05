"""All-190 continuation for VerMo overfit validation.

This resumes from the pretrain-focused checkpoint that passes spt1 greedy, then
returns to the original 19 tasks x 10 samples annotation to eliminate the
remaining full-set teacher-forced misses.
"""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage2_pretrain_focus_lr2e5_from_debug2ckpt200.py'

work_dir = (
    'work_dirs/'
    'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage3_all190_lr5e6_from_focusckpt150'
)

auto_resume = False
load_from = dict(
    _delete_=True,
    path=(
        'work_dirs/'
        'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage2_pretrain_focus_lr2e5_from_debug2ckpt200/'
        'checkpoint-iter_150'
    ),
    load_scope='model',
)

train_dataloader = dict(
    dataset=dict(
        anno_file='data/annotation/vermo_overfit_alltasks_190_20260603.json',
    ),
)

optimizer = dict(
    type='AdamW',
    lr=5e-6,
    betas=[0.9, 0.99],
    weight_decay=0.0,
)

train_cfg = dict(
    by_epoch=False,
    max_iters=500,
    val_interval=10000,
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=1),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=50,
        max_keep_ckpts=10,
        save_last=True,
    ),
)
