# VerMo Pretrain: 16k codebook, Llama-3.2-1B-Instruct backbone
# Launch: bash tools/taiji_dist_train.sh configs/vermo/vermo_pretrain_16k_llama1b_wavtokenizer.py 8

_base_ = './_base_vermo_pretrain_wavtokenizer.py'

# Base defaults match: 16k codebook + Llama-3.2-1B-Instruct. No overrides needed.
