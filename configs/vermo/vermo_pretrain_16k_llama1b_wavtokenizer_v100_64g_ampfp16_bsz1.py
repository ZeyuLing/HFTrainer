"""Full VerMo pretraining on 64 V100 GPUs, batch size 1 per GPU.

This keeps the V100-safe numerical setup validated by overfit:
fp32 trainable LLM parameters under fp16 autocast/GradScaler.  It avoids
gradient accumulation because the current runner steps optimizers every
microbatch, which is unsafe with Accelerate's no_sync accumulation.
"""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer.py'

work_dir = 'work_dirs/vermo_pretrain_16k_llama1b_wavtokenizer_v100_64g_ampfp16_bsz1'

model = dict(
    lm=dict(
        module_dtype='fp32',
    ),
)

accelerator = dict(
    mixed_precision='fp16',
    gradient_accumulation_steps=1,
)

train_dataloader = dict(
    batch_size=1,
)

optimizer = dict(
    lr=3e-5,
)
