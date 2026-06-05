"""Full VerMo pretraining with the intended precision/layout.

This config targets GPUs with native bf16 tensor-core support.  It keeps the
frozen processor / motion-audio tokenizers in fp32, trains the LLM in bf16, and
uses PyTorch SDPA instead of eager attention.  The 8192-token cap is sized for
12s clips with up to three motion streams; 2048 truncates even 12s single-person
motion outputs.
"""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer.py'

work_dir = 'work_dirs/vermo_pretrain_16k_llama1b_wavtokenizer_bf16_sdpa_seq8192'

model = dict(
    processor=dict(
        module_dtype='fp32',
        instruction_stage=True,
        max_seq_len=8192,
    ),
    lm=dict(
        module_dtype='bf16',
        from_pretrained=dict(
            attn_implementation='sdpa',
        ),
    ),
)

accelerator = dict(
    mixed_precision='bf16',
    gradient_accumulation_steps=1,
)

train_dataloader = dict(
    batch_size=1,
)

optimizer = dict(
    lr=3e-5,
)
