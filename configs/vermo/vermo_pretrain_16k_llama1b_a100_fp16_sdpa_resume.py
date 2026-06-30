"""Resume VerMo pretraining on A100 with the PROVEN-stable fp16 numerics but
SDPA (mem-efficient) attention instead of eager.

Validated 2026-06-25 on A100 (torch2.5.0 / transformers4.57.3): with seq2048 +
gradient_checkpointing + autocast + thousands of fully-masked left-pad rows,
SDPA produces bit-identical loss to eager (6.3365 vs 6.3365) with zero NaN, and
is ~3x faster on the attention path (244ms -> 80ms fwd+bwd in the probe).

The historical "eager-only" constraint was a V100 (sm_70, no FlashAttention)
limitation; the earlier A100 "bf16+sdpa" NaN was caused by the bf16 dtype
change, NOT by SDPA.  This config changes ONLY attn_implementation vs the
proven a100 fp16/eager resume config.
"""

_base_ = './vermo_pretrain_16k_llama1b_a100_fp16_eager_resume.py'

work_dir = 'work_dirs/vermo_pretrain_16k_llama1b_a100_fp16_sdpa_resume'

model = dict(
    lm=dict(
        from_pretrained=dict(
            attn_implementation='sdpa',
        ),
    ),
)
