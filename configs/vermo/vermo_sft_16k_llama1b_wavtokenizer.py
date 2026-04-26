# VerMo SFT: 16k codebook, Llama-3.2-1B-Instruct backbone
# Launch: bash tools/taiji_dist_train.sh configs/vermo/vermo_sft_16k_llama1b_wavtokenizer.py 8

_base_ = './_base_vermo_sft_wavtokenizer.py'

# Base defaults match: 16k codebook + Llama-3.2-1B-Instruct. No model overrides needed.
# Pre-migrated checkpoint: work_dirs/vermo_sft_16k_llama1b_wavtokenizer/checkpoint-iter_44000/
# Launch with --auto-resume to continue training from there.
