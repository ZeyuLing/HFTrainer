# Debug Insights
> status: running
> iteration: 3
> best_result: root cause locked — training regression between epoch 657 → 848
> last_updated: 2026-04-20 22:25:00

## 基础设施状态
- 快速验证脚本: ✅ 已创建 (test_quick.py — 单 ckpt 单样本 jitter)
- 数据分类: ✅ 已分析
- baseline: ✅ 旧 epoch_657 数据在 DB

## 重大结论（Iter 1-2）

**根因 = 训练恶化**。用户确认"其他任务没问题"是基于 dashboard 的旧 epoch_657 数据。本轮推理代码改动不是 bug。

**关键证据回顾**:
1. 跨任务同 ckpt × 15 sample: E2/A 9.5×, E3/A 117×, E5/A 6.5×, E9/C_full 16.5× — 全面恶化
2. 跨 ckpt 同代码同 seed: jitter 随 epoch 单调上升 36% (epoch 845→848)
3. E3 比 E9 更严重（117× vs 16.5×），推翻 E9-specific bug 假设

## 新阶段目标
Debug 训练：定位 epoch 657 → 848 之间是什么让模型越训越坏。

## 活跃假设 (新轮)

### H-T1: Loss 曲线实际上在上升或震荡（训练发散）
- 验证方法: 看 wandb/tensorboard 的 uncond_global training log
- 优先级: 最高（决定后续所有方向）

### H-T2: Epoch 657 之后训练代码/config/data 有改动
- 子假设:
  - H-T2a: Dataset transform 最近被改（可能本轮前 1-2 天）
  - H-T2b: Loss weighting / scheduler 改过
  - H-T2c: EMA decay 改过或被重置
- 验证: git log + 对比 config 文件
- 优先级: 高

### H-T3: Checkpoint save 逻辑 bug — 保存的是某个错误状态的权重
- 验证: 看 training logger 的 val loss 曲线 — 如果 val loss 还在下降但 eval jitter 上升，就是 checkpoint 选择不当；如果 val loss 也上升就是真过拟合
- 优先级: 中

### H-T4: LR 太高、训练持续发散（常见于 fine-tune 阶段）
- 验证: 查看 LR schedule 和当前训练 step 的 LR
- 优先级: 中

## 当前观测手段
- test_quick.py 可对多 ckpt 做系统扫描（定位恶化起点）

## 过程观察结论（累积）
1. sliding-window 不是问题
2. SDEdit 不是问题
3. E9-specific 推理 bug 不成立（E3 比 E9 还差）
4. 所有任务都恶化（E2/E3/E5/E9）
5. 连续 ckpt jitter 单调上升 → 训练仍在恶化中
6. uncond_global 的 ADE=0 / foot_float=0.4-0.6 / boundary_accel_jump 上万 等细节暗示**输出是在训练数据分布之外**（高频抖动+位置偏差）

## 待探索方向 (Iter 3+)

1. **查 wandb / tensorboard 的训练 loss** — 优先级: 最高 — 定位 loss 曲线是在下降、还是过拟合、还是发散
2. **git log 最近 commit** 看 training/data pipeline 改动 — 优先级: 高
3. **跨 epoch 系统扫描**：对 uncond_global 跑 epoch 825/830/835/840/845/848 同样本 jitter，定位恶化起始点（如果 wandb 不好用）— 优先级: 高
4. **看 training config**（LR schedule, EMA decay, loss weights） — 优先级: 中
5. **考虑是否有更老的 checkpoint backup** （在别的目录或别的机器上）— 优先级: 中
</content>
