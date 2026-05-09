# E4 根因最终定位（2026-04-22 第二轮）

## 用户报告
- **黄色（generated）帧** 飘浮
- **蓝色（condition）帧** 陷入地下

## 实证数据（uncond_local × E4_A_rhand_sparse × sample 0）

| 帧类型 | foot_minY 平均 | pelvis_Y 平均 | head_Y 平均 | R_Wrist_Y |
|--------|---------------|--------------|------------|-----------|
| Condition (每10帧) | **-0.164m** | +0.703m | +1.260m | +0.822m |
| Generated | +0.004m | +0.869m | +1.427m | +0.978m |
| **差值** | **-0.168m** | **-0.166m** | **-0.167m** | **-0.156m** |

**整个身体在 condition 帧下沉 ~17cm**，不只是脚。这是人物在 cond / gen 帧之间"瞬移"的视觉效果。

## 根因：E4 mask pattern 是 OOD

`build_end_effector_mask` 在 198-dim 下构建的 mask：

Condition 帧（每 10 帧一次）：
```
mask[t, 0:3] = 0          # pelvis 平移 → 用 GT
mask[t, 3+j*6:3+(j+1)*6] = 0  # 仅 R_Wrist (j=21) rot6d → 用 GT
mask[t, 135+(j-1)*3:...] = 0   # R_Wrist 位置通道 → 用 GT
# 其它 21 个关节 rot6d 保持 mask=1 → 模型生成
```

### 为什么是 OOD
对照 CLAUDE.md 训练 mask 策略（M1-M7）：
- M1/M2/M4/M6/M7 在 `(T, 23)` joint-group grid 上操作
- **从未出现过"某一帧只有 pelvis 平移 + R_Wrist 是 condition，其它 21 关节全 generated"的 pattern**
- 更糟：cond 帧（t=10）和相邻 gen 帧（t=11）的 mask pattern **完全不同**，这种 per-frame 切换也不是训练分布

### 模型在 cond 帧的行为
- Pelvis 平移被强制为 GT
- R_Wrist rot6d 被强制为 GT
- 其它 21 关节 rot6d 是**从未见过的 mask 上下文中产生的输出** → 位姿怪异
- 经过 FK（`pelvis_trans + ancestor_rots × bone_offsets`）→ feet 落到 -0.16m（穿地）

### 为什么 generated 帧正常
gen 帧 mask = all-1 = "全帧生成" = **M5 full_mask 训练 pattern** → 模型能正常生成合理姿态 → feet 贴地。

## 后处理也放大问题
`evaluate_sample()` line 2275:
```python
output_135[cond_mask] = motion_135[cond_mask]
```
仅在 mask=0 的 dim 上用 GT 替换 —— R_Wrist rot6d 和 pelvis trans 是 GT，但其它 21 关节 rot6d 还是模型输出。这导致 cond 帧 **FK 后 R_Wrist 世界位置对（metric 好看）**，但姿态严重扭曲。

## 正确修复方案

### Option A：丢弃 per-frame sparse mask，改用 M4 joint-contiguous 训练分布
整段时间都 mask `R_Wrist` 关节组，把 E4 变成"已知 21 个关节 + 已知 pelvis 平移 + 未知 R_Wrist 轨迹，要求满足稀疏约束"。
- ✅ 符合训练分布（M4: joint_contiguous）
- ❌ 改变了 E4 语义（原本只约束稀疏帧的 wrist，现在整段 wrist 都 generated）
- 需要重跑所有 E4

### Option B：Post-hoc FK 校正（轻量）
推理用 M5 full_mask（整段全生成），推理完后在 cond 帧用 IK 把 R_Wrist 拉到目标位置，但对其它关节保留模型的生成结果。
- ✅ 无 OOD（整段 full_mask = M5 训练分布）
- ✅ 符合 E4 原本的稀疏约束语义
- ❌ 需要实现 IK solver（手臂 2-joint IK 简单）

### Option C：切到 ` _man` 变体 + imputation
使用 `_man` 训练的 checkpoint，推理时 `replacement='skip_last'`，clean_motion 提供 cond 帧的 pelvis + R_Wrist。但其它 21 关节没有 clean 信号，仍然要从噪声生成 —— **可能还是 OOD**。

### Option D：降级 cond 帧为 full-pose keyframe（语义变了）
在 cond 帧把**所有关节**设为 condition（整帧 pose 锁 GT），这就变成 "keyframe inbetweening" (E2)，不是 "end-effector constraint" (E4)。

## 推荐
**Option B**：最小改动 + 保留 E4 原始语义 + 符合训练分布。实现步骤：
1. `build_end_effector_mask` 改为全 1 mask（M5 full_mask），`constraint_info` 保留
2. 推理正常跑
3. `evaluate_sample` 后处理：
   - FK → world positions
   - 对 cond 帧的 target joint (R_Wrist) 位置计算 error（metric 不变）
   - 可选：2-joint arm IK 把 R_Wrist 真正拉到 target 位置再输出 NPZ（视觉上正确）

## 可视化修复应该回滚吗？
可以回滚 `groundOffset = 0` 改动，因为现在 pred 烂的不是"整个 motion 下沉"而是"cond 帧下沉 17cm，gen 帧正常"。用原来的 `-pred_min_Y` 让 gen 帧贴地，但 cond 帧在 canonicalizeGround 里**整段 minY = -0.433** 会被用于归一化 —— 等于 cond 帧高度被抬 43cm，gen 帧就整体飘浮。

**唯一真正的修复是重构 E4 推理**，不是 viz。
