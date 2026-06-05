"""Low-LR continuation from the best strict overfit checkpoint so far.

Iter 9500 passes the text/motion multi-person quick gate (T2M/M2T including
true and pseudo multi-person cases), but the full 180-task export still shows
non-exact audio/speech-related outputs.  Continue from that stable point with a
smaller LR while keeping the same deterministic 180-case dataset.
"""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_overfit_mainpipeline_multimode_180_deterministic.py'

work_dir = (
    'work_dirs/'
    'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_mainpipeline_multimode_180_deterministic_lr5e6_from9500'
)

auto_resume = False
load_from = dict(
    _delete_=True,
    path=(
        'work_dirs/'
        'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_mainpipeline_multimode_180_deterministic/'
        'checkpoint-iter_9500'
    ),
    load_scope='full',
)

optimizer = dict(lr=5e-6)

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
