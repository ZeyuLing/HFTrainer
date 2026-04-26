# VerMo T2M+M2T: 16k codebook, Qwen3-0.6B backbone
# Launch: bash tools/taiji_dist_train.sh configs/vermo/vermo_t2m_m2t_16k_qwen0.6b_wavtokenizer.py 8

_base_ = './_base_vermo_t2m_m2t_wavtokenizer.py'

# Base defaults match: 16k codebook + Qwen3-0.6B. No model overrides needed.

load_from = dict(
    path='work_dirs/vermo_t2m_pretrain_qwen0.6b_16k_hq/iter_20000.pth',
    load_scope='model',
)
