"""Strict 180-case overfit after fixing true-multi caption conditioning.

True multi-person samples must use the union caption as the Caption modal.
Per-person captions are still loaded as metadata, but they are not fed as the
caption input/target for true multi tasks.  Start from the closest previous
overfit checkpoint and re-fit the changed text targets/prompts.
"""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_overfit_mainpipeline_multimode_180_deterministic.py'

work_dir = (
    'work_dirs/'
    'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_mainpipeline_multimode_180_deterministic_unioncaption_lr5e6_from10775'
)

auto_resume = False
load_from = dict(
    _delete_=True,
    path=(
        'work_dirs/'
        'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_mainpipeline_multimode_180_deterministic_lr1e6_from10750/'
        'checkpoint-iter_10775'
    ),
    load_scope='full',
)

optimizer = dict(lr=5e-6)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=1),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=25,
        max_keep_ckpts=20,
        save_last=True,
    ),
)
