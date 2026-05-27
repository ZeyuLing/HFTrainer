status: running

# 活跃结论
- 根因高置信：`spectral_unified` 使用 `np.linalg.norm(spectral_coords, axis=1)`，把多维 spectral coordinate 压成无符号半径，造成对称关节 RoPE position 碰撞。
- 训练 loss 稳定下降但 eval 异常，说明不是 NaN/崩溃/普通 checkpoint persistence 问题。

# 证据
- overfit_100 主训练 run epoch 1224: loss≈0.0493, loss_rot≈0.0930。
- eval_epoch99 mean_l2_error≈2.28，epoch299_nocfg mean_l2_error≈2.17。
- 谱位置唯一性检查：22 body joints 只有 14 个唯一 rounded scalar positions，左右 limb 成对重复。

# 下一步建议
- 不要继续使用当前 `spectral_unified`。
- 修复方案 A：改为不会碰撞的 signed scalar projection，并加入唯一性单测。
- 修复方案 B：直接用 `dfs` 作为保守 topology baseline。
- 修复方案 C：重新设计真正 multi-dimensional spectral RoPE，同时处理预训练兼容问题。
