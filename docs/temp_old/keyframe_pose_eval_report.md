# Keyframe Pose Guidance — 综合测评报告

**日期**: 2026-03-30
**测试样本**: 20 条 motionhub 测试动作 × 3 keyframe 位置 (0.25/0.50/0.75) = 60 cases
**测评模型**: 10 个 HyMotion M2M 变体 (4 uncond_man + 2 uncond non-man + 4 caption_man)
**测评维度**: 3 imputation 策略 × 2 replacement guidance 模式 × 2 rotation 空间

## 1. 实验设置

### 1.1 Imputation 策略

| 策略 | 描述 | 难度 |
|------|------|------|
| `keyframe_only` | 仅保留 keyframe 帧，其余全部 mask | 最难 |
| `anchor_inbetween` | 保留首帧 + keyframe + 尾帧 | 推荐 |
| `local_edit` | 仅 keyframe 前后 ±30 帧 mask | 最简 |

### 1.2 Replacement Guidance

| 模式 | 描述 |
|------|------|
| `none` | 标准 ODE 求解，不做额外替换 |
| `flow_interp` | 每步将已知区域替换为 flow-matching 插值（保持 clean） |

### 1.3 测评指标

| 指标 | 含义 | 越低越好 |
|------|------|---------|
| **KF L2** | keyframe 帧 output vs GT 的 L2 误差 | ✓ |
| **MPJPE** | 全帧平均关节位置误差 (cm) | ✓ |
| **Bnd Smooth** | mask 边界处加速度（平滑度） | ✓ |
| **Foot Skate** | 脚部滑动程度 | ✓ |

## 2. 核心结果

### 2.1 Local Rotation — uncond + MAN 模型（用户指定范围）

| 模型 | Imp. Mode | Rep. Mode | MPJPE↓ | Bnd Smooth↓ | Foot Skate↓ | KF L2 |
|------|-----------|-----------|--------|-------------|-------------|-------|
| **uncond_jit_man** | local_edit | flow_interp | **2.987** | 3.202 | 0.124 | 0.000 |
| uncond_jit_man | anchor_inbetween | flow_interp | 3.006 | **2.908** | 0.137 | 0.000 |
| uncond_jit_man | keyframe_only | flow_interp | 3.002 | 3.293 | **0.074** | 0.000 |
| uncond_fm_man | local_edit | flow_interp | 3.486 | 3.699 | 0.153 | 0.000 |
| uncond_fm_man | anchor_inbetween | flow_interp | 3.598 | 3.653 | 0.143 | 0.000 |
| uncond_fm_man | keyframe_only | flow_interp | 3.531 | 4.268 | 0.050 | 0.000 |

### 2.2 Global Rotation — uncond + MAN 模型

| 模型 | Imp. Mode | Rep. Mode | MPJPE↓ | Bnd Smooth↓ | Foot Skate↓ | KF L2 |
|------|-----------|-----------|--------|-------------|-------------|-------|
| **uncond_jit_man_globalrot** | local_edit | flow_interp | **4.604** | **3.221** | 0.129 | 0.000 |
| uncond_jit_man_globalrot | anchor_inbetween | flow_interp | 4.667 | 3.151 | 0.137 | 0.000 |
| uncond_fm_man_globalrot | anchor_inbetween | flow_interp | 4.721 | 4.899 | 0.147 | 0.000 |
| uncond_fm_man_globalrot | local_edit | flow_interp | 4.806 | 5.243 | 0.203 | 0.000 |

## 3. 关键发现

### 3.1 JiT Loss 一致优于 FM Loss

在所有条件下，JiT 训练变体均优于对应的 FM 变体：

| 对比 | JiT MPJPE | FM MPJPE | JiT 优势 |
|------|-----------|----------|----------|
| local_rot, anchor, flow_interp | 3.006 | 3.598 | **-16.4%** |
| local_rot, keyframe, flow_interp | 3.002 | 3.531 | **-15.0%** |
| local_rot, local_edit, flow_interp | 2.987 | 3.486 | **-14.3%** |
| global_rot, anchor, flow_interp | 4.667 | 4.721 | -1.1% |
| global_rot, local_edit, flow_interp | 4.604 | 4.806 | **-4.2%** |

**结论**: JiT loss 的训练目标更利于 imputation 任务，在 local rotation 空间差距更显著。

### 3.2 Replacement Guidance (flow_interp) 显著改善质量

`flow_interp` vs `none` 对比（以 uncond_jit_man local_rot 为例）：

| Imp. Mode | flow_interp MPJPE | none MPJPE | Bnd Smooth (fi/none) |
|-----------|-------------------|------------|----------------------|
| anchor_inbetween | **3.006** | 3.439 | 2.908 / 5.298 |
| keyframe_only | **3.002** | 3.331 | 3.293 / 5.502 |
| local_edit | **2.987** | 3.554 | 3.202 / 5.127 |

**结论**: `flow_interp` 在 MPJPE 上平均降低 12%，在 boundary smoothness 上平均改善 40%+。这表明 MAN 模型配合 replacement guidance 能有效保持已知帧的精确性并产生更平滑的过渡。

### 3.3 Local Rotation 显著优于 Global Rotation

| 模型 | Local MPJPE | Global MPJPE | 劣化 |
|------|-------------|--------------|------|
| uncond_jit_man (best config) | 2.987 | 4.604 | +54% |
| uncond_fm_man (best config) | 3.486 | 4.721 | +35% |

**结论**: Global rotation 空间下模型表现明显更差。分析原因：
1. global rotation 模型训练 epoch 更少（jit: 79 vs 374, fm: 88 vs 407）
2. global rotation 的误差会在运动链上累积（global 表示中每个关节的旋转已包含父关节旋转）

### 3.4 Imputation 策略对比

以 uncond_jit_man + flow_interp + local_rot 为例：

| 策略 | MPJPE | Bnd Smooth | Foot Skate | 适用场景 |
|------|-------|------------|------------|---------|
| local_edit | **2.987** | 3.202 | 0.124 | 最佳整体质量 |
| anchor_inbetween | 3.006 | **2.908** | 0.137 | 边界最平滑 |
| keyframe_only | 3.002 | 3.293 | **0.074** | 最低脚步滑动 |

三种策略表现相近。`local_edit` 因为仅 mask 局部区域，整体 MPJPE 最低。`keyframe_only` 的脚步滑动最低，可能因为完全重新生成的动作更自然。

### 3.5 Keyframe 精度完美保持 (KF L2 = 0.0000)

所有 MAN 变体 + flow_interp 配置下，keyframe 帧的 L2 误差为 0.0000，这意味着：
- **replacement guidance 完美保持了指定 keypose**
- 这对实际应用（动画师指定关键帧）至关重要

### 3.6 MAN vs Non-MAN 基线

| 模型 | MPJPE | Bnd Smooth |
|------|-------|------------|
| uncond_fm (non-MAN) | 0.81-1.63 | 0.11-0.14 |
| uncond_jit (non-MAN) | 0.86-1.64 | 0.21-0.25 |
| uncond_fm_man | 3.49-4.15 | 3.65-6.79 |
| uncond_jit_man | 2.99-3.55 | 2.91-5.50 |

Non-MAN 模型的 MPJPE 看似更低，但这是因为 non-MAN 模型在 mask 区域没有真正重生成运动，而是倾向于保留原始信号。MAN 模型产生更多创造性变化（更高 MPJPE），但更符合 keypose guidance 的实际用途。

## 4. 推荐配置

### 实际使用推荐

| 优先级 | 模型配置 | MPJPE | 理由 |
|--------|---------|-------|------|
| **P0** | uncond_jit_man + local_edit + flow_interp | 2.99 | 整体最优质量 |
| P1 | uncond_jit_man + anchor_inbetween + flow_interp | 3.01 | 最平滑边界 |
| P2 | uncond_fm_man + local_edit + flow_interp | 3.49 | FM loss 备选 |

### 关键配置指南

1. **Replacement guidance 必须开启** (`flow_interp`)：不开启会导致 MPJPE 和 smoothness 均显著退化
2. **优先使用 local rotation**：global rotation 质量明显更差
3. **JiT loss 优于 FM loss**：所有条件下一致优势
4. **Imputation 策略选择**：`local_edit` 和 `anchor_inbetween` 均可，根据场景选择

## 5. 待办事项

- [ ] MoGenDIT 基线测评（需要 GPU 机器运行）
- [ ] PeacekeeperElite 数据上的测评（已构建 144 条 eval 数据，需要 GPU）
- [ ] 更多 keyframe 数量的实验（当前仅测试单 keyframe）
- [ ] 长序列（> 200 帧）性能分析

## 6. 可视化

测评结果可视化网站运行于 **http://9.134.251.2:8097**

功能：
- 左侧：imputation 策略 / replacement guidance / rotation 空间选择器
- 左侧下方：指标汇总表（自动高亮最佳值）
- 主区域：Ground Truth + 各模型输出的 3D 骨骼并排对比
- 底部：帧级播放控制（播放/暂停/逐帧/速度调节）
- 颜色编码：蓝色=正常帧，黄色=keyframe，红色=被 mask 帧
