"""DHA copy of the accumulated stage-2 VerMo overfit validation."""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage2_outputonly_lr1e5_1gpu_accum8_skipguard.py'

work_dir = (
    'work_dirs/'
    'vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage2_outputonly_lr1e5_from1250_1gpu_accum8_dha_skipguard'
)
