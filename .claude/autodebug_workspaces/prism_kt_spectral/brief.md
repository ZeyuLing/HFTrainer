# PRISM KT spectral overfit 异常调试简报

## 问题
PRISM KT spectral/spectral_unified 版本在 100 sample overfit 训练集上训练 loss 下降，但生成/eval 明显异常。

## 关键代码
- `configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py` 将 transformer 设置为 `joint_pos_mode="spectral_unified"`。
- `hftrainer/models/motion/prism/network/motion_rope.py` 的 `spectral_unified` 使用 4 维 Laplacian spectral coords 的 L2 norm 作为每个 joint 的标量 RoPE position。

## 远端环境
- lzy_debug_machine_1/2 均为 1x8 V100，运行中。
- 远端当前 repo commit: c4e76f1。
- 远端存在 `work_dirs/prism_overfit_100` 与 PRISM t5cached v5-v9 workdirs。
