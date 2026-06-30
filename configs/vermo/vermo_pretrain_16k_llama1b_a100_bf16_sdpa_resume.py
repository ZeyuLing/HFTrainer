"""Resume VerMo pretraining on 32x A100 (4 nodes x 8) with bf16 + SDPA.

Warm-starts the model weights from the latest V100 fp16 checkpoint
(``iter_204000``); the optimizer / step counter reset (safe here: constant LR,
no scheduler, AdamW momentum re-warms in a few hundred steps).  The training
*objective is unchanged* — only precision (bf16), attention kernel (SDPA),
sequence cap (8192), and per-GPU batch are upgraded for the A100 80GB cards.

Launch (per node, NODE_RANK=0..3, MASTER_ADDR=<launcher ip>):
    NNODES=4 NODE_RANK=<r> MASTER_ADDR=<ip> MASTER_PORT=29501 \
        bash tools/dist_train.sh \
        configs/vermo/vermo_pretrain_16k_llama1b_a100_bf16_sdpa_resume.py 8
"""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_bf16_sdpa_seq8192.py'

work_dir = 'work_dirs/vermo_pretrain_16k_llama1b_a100_bf16_sdpa_resume'

# First launch warm-starts from the V100 checkpoint (model weights only).
# After the first A100 checkpoint lands in work_dir, auto_resume takes over so
# preemption / restarts continue from this run rather than re-warming.
auto_resume = True
load_from = dict(
    _delete_=True,
    load_scope='model',
    path=(
        'work_dirs/vermo_pretrain_16k_llama1b_wavtokenizer_v100_64g_ampfp16_bsz1_seq2048_eager/'
        'checkpoint-iter_204000'
    ),
)

# A100 80GB: lift per-GPU batch (32 GPUs x 2 = effective batch 64, 2x the V100 run).
train_dataloader = dict(
    batch_size=2,
)
