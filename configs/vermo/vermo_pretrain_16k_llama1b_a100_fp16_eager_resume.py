"""Resume VerMo pretraining on 32x A100 (4 nodes x 8) with the PROVEN-stable
V100 numerics (fp16 autocast + fp32 LM master + eager attention), warm-started
from iter_204000.

Rationale: the bf16 + SDPA config produced loss=nan from step 1 on this model
(SDPA was already documented to fail for this model on the V100 attempts, and
the bf16 path is unvalidated).  This config keeps the exact numerics that gave
the stable ~3.0 loss and only upgrades the hardware (A100) and per-GPU batch.
Objective is unchanged.
"""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_v100_64g_ampfp16_bsz1_seq2048_eager.py'

work_dir = 'work_dirs/vermo_pretrain_16k_llama1b_a100_fp16_eager_resume'

# Warm-start model weights from the latest V100 checkpoint; optimizer/step reset
# (safe: constant LR, no scheduler).  After the first A100 checkpoint lands in
# work_dir, auto_resume continues from this run on preemption/restart.
auto_resume = True
load_from = dict(
    _delete_=True,
    load_scope='model',
    path=(
        'work_dirs/vermo_pretrain_16k_llama1b_wavtokenizer_v100_64g_ampfp16_bsz1_seq2048_eager/'
        'checkpoint-iter_204000'
    ),
)

# A100 80GB: double per-GPU batch (32 GPUs x 2 = effective batch 64, 2x V100).
train_dataloader = dict(
    batch_size=2,
)
