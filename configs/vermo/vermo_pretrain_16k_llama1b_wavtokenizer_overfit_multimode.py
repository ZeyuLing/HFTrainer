"""VerMo 16k Llama-1B multimode overfit validation config.

The annotation contains 18 tasks x 10 deterministic cases.  Non-audio
text/motion tasks include true and pseudo two-person cases; audio/music/speech
tasks stay single-person unless the source data is genuinely multi-person.
Train and eval use the same annotation.
"""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks.py'

work_dir = 'work_dirs/vermo_pretrain_16k_llama1b_wavtokenizer_overfit_multimode_180_short2s_promptuniq_textpseudo_noaudio_keepoutput2048'

auto_resume = False
load_from = None

model = dict(
    processor=dict(
        instruction_stage=True,
        max_seq_len=2048,
    ),
    lm=dict(
        from_pretrained=dict(
            attn_implementation='eager',
        ),
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
    dataset=dict(
        anno_file='data/annotation/vermo_overfit_multimode_180_short2s_promptuniq_textpseudo_noaudio_20260604.json',
        task_mode='preset',
        preset_tasks=vermo_tasks_no_pretrain,
        task_bucket_mode='none',
        log_task_iter=10,
    ),
)
