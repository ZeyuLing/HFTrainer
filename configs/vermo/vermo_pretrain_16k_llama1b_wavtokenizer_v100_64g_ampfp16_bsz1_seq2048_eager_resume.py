"""Resume full VerMo pretraining on 64 V100 GPUs from the latest eager checkpoint."""

_base_ = './vermo_pretrain_16k_llama1b_wavtokenizer_v100_64g_ampfp16_bsz1_seq2048_eager.py'

auto_resume = True
load_from = None
