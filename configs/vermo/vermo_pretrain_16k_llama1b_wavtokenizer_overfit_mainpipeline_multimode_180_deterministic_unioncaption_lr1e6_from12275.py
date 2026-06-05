"""Low-LR continuation for the union-caption 180-case overfit.

The 5e-6 union-caption run reaches very low loss but shows checkpoint-to-
checkpoint generation jitter.  Continue from a hardlinked 12275 checkpoint
snapshot with 1e-6 to settle the remaining exact-match misses.
"""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_overfit_mainpipeline_multimode_180_deterministic.py'

work_dir = (
    'work_dirs/'
    'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_mainpipeline_multimode_180_deterministic_unioncaption_lr1e6_from12275'
)

auto_resume = False
load_from = dict(
    _delete_=True,
    path='work_dirs/vermo_seed_overfit_det180_unioncaption_checkpoint_iter_12275',
    load_scope='full',
)

optimizer = dict(lr=1e-6)

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
