"""Manual debug-machine stage-2 continuation from the 1-GPU ckpt500."""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks.py'

work_dir = (
    'work_dirs/'
    'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage2_outputonly_lr1e5_from_stage2ckpt500_debug2'
)

auto_resume = False
load_from = dict(
    _delete_=True,
    path=(
        'work_dirs/'
        'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage2_outputonly_lr2e5_from1250_1gpu_skipguard/'
        'checkpoint-iter_500'
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

accelerator = dict(
    mixed_precision='fp16',
    gradient_accumulation_steps=1,
)

optimizer = dict(
    type='AdamW',
    lr=1e-5,
    betas=[0.9, 0.99],
    weight_decay=0.0,
)

train_cfg = dict(
    by_epoch=False,
    max_iters=1200,
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
