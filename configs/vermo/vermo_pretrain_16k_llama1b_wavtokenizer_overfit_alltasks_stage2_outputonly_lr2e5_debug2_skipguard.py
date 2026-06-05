"""Manual debug-machine copy of the 8-GPU stage-2 VerMo overfit validation."""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage2_outputonly_lr2e5.py'

work_dir = (
    'work_dirs/'
    'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage2_outputonly_lr2e5_from1250_debug2_skipguard'
)
