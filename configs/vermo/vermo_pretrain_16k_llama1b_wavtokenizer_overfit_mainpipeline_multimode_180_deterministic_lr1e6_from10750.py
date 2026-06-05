"""Very-low-LR continuation from the closest strict overfit checkpoint.

Iter 10750 already passes every deterministic 180-case task except two single
T2M/N2TM token positions.  This continuation keeps the exact same overfit data
and main inference/training pipeline, but uses a smaller LR and denser
checkpoints so we can search for a fully exact greedy-inference checkpoint
without disturbing the already-correct multi-person and audio/speech cases.
"""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_overfit_mainpipeline_multimode_180_deterministic.py'

work_dir = (
    'work_dirs/'
    'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_mainpipeline_multimode_180_deterministic_lr1e6_from10750'
)

auto_resume = False
load_from = dict(
    _delete_=True,
    path=(
        'work_dirs/'
        'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_mainpipeline_multimode_180_deterministic/'
        'checkpoint-iter_10750'
    ),
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
