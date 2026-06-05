"""Elastic 64-GPU fallback for the short2s multimode VerMo overfit validation."""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_overfit_multimode.py'

work_dir = 'work_dirs/vermo_pretrain_16k_llama1b_wavtokenizer_overfit_multimode_180_short2s_promptuniq_textpseudo_noaudio_keepoutput2048_elastic64g'
