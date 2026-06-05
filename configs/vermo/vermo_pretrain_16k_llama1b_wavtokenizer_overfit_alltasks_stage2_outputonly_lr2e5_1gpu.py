"""Single-GPU stage-2 VerMo all-task overfit validation."""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage2_outputonly_lr2e5.py'

work_dir = (
    'work_dirs/'
    'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage2_outputonly_lr2e5_from1250_1gpu'
)

train_cfg = dict(
    by_epoch=False,
    max_iters=3000,
    val_interval=10000,
    max_grad_norm=1.0,
)
