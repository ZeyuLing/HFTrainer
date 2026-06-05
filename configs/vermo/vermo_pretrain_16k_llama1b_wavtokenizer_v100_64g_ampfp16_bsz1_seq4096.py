"""Full VerMo pretraining on 64 V100 GPUs with a 4096-token LM cap.

The uncapped full-data run hit a CUDA invalid-argument error inside Llama SDPA
on early long batches.  Keep the V100-safe batch/numerics from the bsz1 config
and cap LM sequence length to avoid pathological attention shapes on V100.
"""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_v100_64g_ampfp16_bsz1.py'

work_dir = 'work_dirs/vermo_pretrain_16k_llama1b_wavtokenizer_v100_64g_ampfp16_bsz1_seq4096'

model = dict(
    processor=dict(
        max_seq_len=4096,
    ),
)
