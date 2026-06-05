"""Full VerMo pretraining on 64 V100 GPUs.

The base config was written with bf16 LLM weights and no accelerator mixed
precision. V100 does not support bf16 efficiently, while the overfit
validation converged cleanly with fp32 LLM weights under fp16 autocast.
"""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer.py'

work_dir = 'work_dirs/vermo_pretrain_16k_llama1b_wavtokenizer_v100_64g_ampfp16'

model = dict(
    lm=dict(
        module_dtype='fp32',
    ),
)

accelerator = dict(
    mixed_precision='fp16',
    gradient_accumulation_steps=2,
)

train_dataloader = dict(
    batch_size=1,
)
