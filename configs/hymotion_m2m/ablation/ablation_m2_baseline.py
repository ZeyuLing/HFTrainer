# Ablation M2: Baseline M2M (control group, 20 epochs)
# 这是 Baseline-M2M 从 HY-Motion-1.0-Lite 续训 20 epoch 的对照组。
# 所有其他消融实验的训练也是 20 epoch，因此需要这个对照组来公平对比。

_base_ = '../hymotion_m2m_completion_uncond_fm_046b.py'

work_dir = 'work_dirs/ablation_m2_baseline'

train_cfg = dict(max_epochs=20)
