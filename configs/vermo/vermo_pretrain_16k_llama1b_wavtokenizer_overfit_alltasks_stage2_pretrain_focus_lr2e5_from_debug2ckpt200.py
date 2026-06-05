"""Pretrain-focused continuation for the VerMo overfit validation."""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks.py'

work_dir = (
    'work_dirs/'
    'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage2_pretrain_focus_lr2e5_from_debug2ckpt200'
)

auto_resume = False
load_from = dict(
    _delete_=True,
    path=(
        'work_dirs/'
        'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage2_outputonly_lr1e5_from_stage2ckpt500_debug2/'
        'checkpoint-iter_200'
    ),
    load_scope='model',
)

model = dict(
    processor=dict(
        instruction_stage=True,
        optional_input_modal_mode='all',
        task_template_mode='first',
        shuffle_modal_parts=False,
        max_seq_len=0,
    ),
    lm=dict(
        module_dtype='fp32',
    ),
)

train_dataloader = dict(
    dataset=dict(
        anno_file='data/annotation/vermo_overfit_alltasks_200_pretrain_focus_20260603.json',
    ),
)

accelerator = dict(
    mixed_precision='fp16',
    gradient_accumulation_steps=1,
)

optimizer = dict(
    type='AdamW',
    lr=2e-5,
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
        max_keep_ckpts=6,
        save_last=True,
    ),
)
