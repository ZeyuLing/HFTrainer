# Ablation T1: EMA (Exponential Moving Average)
# 验证：EMA 是否能提高生成质量的稳定性。
# KIMODO 使用 EMA decay=0.995, every 10 steps。
#
# 改动：+EMAHook(decay=0.995, update_interval=10)

_base_ = '../hymotion_m2m_completion_uncond_fm_046b.py'

work_dir = 'work_dirs/ablation_t1_ema'

train_cfg = dict(max_epochs=20)

default_hooks = dict(
    ema=dict(type='EMAHook', decay=0.995, update_interval=10),
)
