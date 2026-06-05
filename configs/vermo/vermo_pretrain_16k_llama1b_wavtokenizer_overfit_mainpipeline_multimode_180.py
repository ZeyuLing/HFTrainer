"""Main-pipeline full-multimode VerMo overfit validation.

This keeps the same model/config inheritance chain as the 64-card main
experiment, but swaps in the deterministic 180-case annotation used for
complete overfit validation:

- 18 tasks x 10 cases.
- T2M/M2T/N2TM/Pred/Inbetween each include single, true multi-person, and
  materialized pseudo multi-person cases.
- Audio/music/speech tasks remain single-person unless the source is genuinely
  multi-person, matching the no-pseudo-audio policy.

The existing main-pipeline 93-case overfit relies on online ComposeMultiPerson,
so the viewer cannot deterministically cover pseudo-multi branches.  This
config makes those branches explicit while preserving the main model,
processor, tokenizer, and preprocessing path.
"""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_v100_64g_ampfp16_bsz1_seq2048_eager.py'

work_dir = (
    'work_dirs/'
    'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_mainpipeline_multimode_180'
)

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

vermo_tasks_no_pretrain = [
    't2m',
    'm2t',
    'n2tm',
    'pred',
    'inbetween',
    'm2d',
    'd2m',
    't2md',
    'g2md',
    'n2md',
    'm2d_ar',
    'd2m_ar',
    's2g',
    'g2s',
    't2sg',
    'n2sg',
    'ss2sg',
    's2g_ar',
]

train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    shuffle=True,
    dataset=dict(
        anno_file=(
            'data/annotation/'
            'vermo_overfit_multimode_180_short2s_promptuniq_textpseudo_noaudio_20260604.json'
        ),
        task_mode='preset',
        preset_tasks=vermo_tasks_no_pretrain,
        task_bucket_mode='none',
        log_task_iter=20,
    ),
)
