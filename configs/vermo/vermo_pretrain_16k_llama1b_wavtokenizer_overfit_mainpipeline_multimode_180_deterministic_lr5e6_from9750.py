"""Low-LR continuation for strict 180-case VerMo overfit.

The high-LR deterministic overfit gets very close around iter 9000-9750, but
the greedy T2M pseudo-multi case can oscillate by one token or terminate early.
This continuation keeps the exact same deterministic dataset/pipeline and uses
a smaller LR to settle the autoregressive outputs.
"""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_overfit_mainpipeline_multimode_180_deterministic.py'

work_dir = (
    'work_dirs/'
    'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_mainpipeline_multimode_180_deterministic_lr5e6_from9750'
)

auto_resume = False
load_from = dict(
    _delete_=True,
    path=(
        'work_dirs/'
        'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_mainpipeline_multimode_180_deterministic/'
        'checkpoint-iter_9750'
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
