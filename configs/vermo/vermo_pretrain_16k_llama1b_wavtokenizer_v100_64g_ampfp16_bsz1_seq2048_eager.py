"""Full VerMo pretraining on 64 V100 GPUs with eager Llama attention.

The previous V100 64-card attempts reached the training loop but failed inside
PyTorch SDPA on early full-data batches.  This keeps the overfit-validated V100
numerics and batch size, caps LM sequence length to fit eager attention in
32GB, and routes Llama away from the SDPA kernel path.
"""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_v100_64g_ampfp16_bsz1.py'

work_dir = 'work_dirs/vermo_pretrain_16k_llama1b_wavtokenizer_v100_64g_ampfp16_bsz1_seq2048_eager'

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
