"""Main-pipeline VerMo overfit config.

This config intentionally inherits the full 64-card eager pretrain config so
the model, task assignment, preprocessing pipeline, ComposeMultiPerson setting,
and tokenization path match the main experiment.  Only the annotation, work_dir,
and checkpoint frequency are reduced for overfit debugging.
"""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_v100_64g_ampfp16_bsz1_seq2048_eager.py'

work_dir = 'work_dirs/vermo_pretrain_16k_llama1b_wavtokenizer_overfit_mainpipeline_18tasks_93_outputstage'

auto_resume = False
load_from = dict(
    _delete_=True,
    path=(
        'work_dirs/vermo_pretrain_16k_llama1b_wavtokenizer_v100_64g_ampfp16_bsz1_seq2048_eager/'
        'checkpoint-iter_8000'
    ),
    load_scope='full',
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=1),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=250,
        max_keep_ckpts=5,
        save_last=True,
    ),
)

train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    shuffle=True,
    dataset=dict(
        anno_file='data/annotation/vermo_overfit_mainpipeline_18tasks_93_20260604.json',
        task_mode='auto',
        log_task_iter=20,
    ),
)
