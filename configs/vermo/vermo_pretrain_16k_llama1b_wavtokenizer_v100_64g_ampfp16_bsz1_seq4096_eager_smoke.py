"""V100 eager-attention baseline for the seq4096 smoke comparison."""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_v100_64g_ampfp16_bsz1.py'

work_dir = (
    'work_dirs/'
    'vermo_pretrain_16k_llama1b_wavtokenizer_v100_64g_ampfp16_bsz1_'
    'seq4096_eager_smoke'
)

model = dict(
    processor=dict(
        module_dtype='fp32',
        instruction_stage=True,
        max_seq_len=4096,
    ),
    lm=dict(
        module_dtype='fp32',
        from_pretrained=dict(
            attn_implementation='eager',
        ),
    ),
)

accelerator = dict(
    mixed_precision='fp16',
    gradient_accumulation_steps=1,
)

train_cfg = dict(
    _delete_=True,
    by_epoch=False,
    max_iters=200,
    val_interval=1000,
)

default_hooks = dict(
    logger=dict(interval=1, iter_interval=1),
    checkpoint=dict(interval=100, max_keep_ckpts=2, save_last=True),
)
