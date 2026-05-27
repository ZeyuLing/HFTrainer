## Iteration 1 - 2026-05-27

### 预注册
假设：异常不是 buffer persistence 或训练 crash，而是 KT spectral positional encoding 本身丢失 joint identity，导致 overfit 训练 loss 可下降但生成/eval 不正常。

### 观察
1. `work_dirs/prism_overfit_100/20260526_212303/train.log` 到 epoch 1224 仍稳定，末段 loss 约 0.049-0.06，loss_rot 约 0.09-0.12。
2. `work_dirs/prism_overfit_100/eval_epoch*.json` 的生成误差仍明显偏大：epoch49 mean_l2≈1.53，epoch74≈2.35，epoch99≈2.28，epoch299_nocfg≈2.17。
3. `spectral_unified` 将 spectral coords 取 L2 norm 后，左右对称关节产生完全相同 RoPE position。

### 关键数值
重复位置包括：L/R Hip、L/R Knee、L/R Ankle、L/R Foot、L/R Collar、L/R Shoulder、L/R Elbow、L/R Wrist。22 个 body joints 只有 14 个唯一标量位置。

### 结论
根因定位到 `spectral_unified` 的 scalarization：L2 norm 去掉了谱坐标方向和符号，破坏左右/分支 identity。
