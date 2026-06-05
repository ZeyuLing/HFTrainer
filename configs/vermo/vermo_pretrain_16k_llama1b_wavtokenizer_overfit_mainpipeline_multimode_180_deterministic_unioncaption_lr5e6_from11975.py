"""Strict 180-case overfit after fixing true-multi caption conditioning.

True multi-person samples must use the union caption as the Caption modal.
Per-person captions are still loaded as metadata, but they are not fed as the
caption input/target for true multi tasks.

This config resumes from a hardlinked checkpoint snapshot so the source
checkpoint cannot disappear while older overfit jobs rotate max_keep checkpoints.
"""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_overfit_mainpipeline_multimode_180_deterministic.py'

work_dir = (
    'work_dirs/'
    'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_mainpipeline_multimode_180_deterministic_unioncaption_lr5e6_from11975'
)

auto_resume = False
load_from = dict(
    _delete_=True,
    path='work_dirs/vermo_seed_overfit_det180_lr1e6_checkpoint_iter_11975',
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
