# Ablation T2 Phase 2: Curriculum Training — Phase 2 (M1-M6 completion)
# Phase 2: 从 Phase 1 checkpoint 续训 20 epoch，标准 M1-M6 混合策略。
#
# 使用方法：
#   先训完 Phase 1 (ablation_t2_curriculum_p1.py)
#   然后运行此 config，需要手动设置 load_from:
#     python tools/train.py configs/hymotion_m2m/ablation/ablation_t2_curriculum_p2.py \
#       --cfg-options load_from.path=work_dirs/ablation_t2_curriculum_p1/checkpoint-epoch_20

_base_ = '../hymotion_m2m_completion_uncond_fm_046b.py'

work_dir = 'work_dirs/ablation_t2_curriculum_p2'

train_cfg = dict(max_epochs=20)

# Load Phase 1 checkpoint (model-only, not full resume)
load_from = dict(
    _delete_=True,
    path='work_dirs/ablation_t2_curriculum_p1/checkpoint-epoch_20',
    load_scope='model',
)

# Standard M1-M6 mask strategy (same as Baseline-M2M)
# Pipeline inherited from base config — uses default M1-M6 weights
