# Ablation L1: FK Loss (Forward Kinematics Joint Position Loss)
# 验证：加入 FK 后 3D 关节位置约束是否能改善 MPJPE。
# KIMODO 的 FK loss 权重 γ_pos=10，是所有分量中最高的。
#
# 改动：keypoints3d_weight: 0 → 0.1
# 需要 body_model_path 以计算 FK

_base_ = '../hymotion_m2m_completion_uncond_fm_046b.py'

work_dir = 'work_dirs/ablation_l1_fk_loss'

train_cfg = dict(max_epochs=20)

model = dict(
    losses_cfg=dict(
        keypoints3d_weight=0.1,
    ),
    body_model_path='ref_repo/MoGenDiT/motion_process/body_model/smplh',
)
