# Keyframe Pose Guidance — 测评方案

## 1. 数据对构建规则

### 1.1 数据来源

`data/PeacekeeperElite_MB/PeacekeeperElite_part4_before_MB` (src) 和
`data/PeacekeeperElite_MB/PeacekeeperElite_part4_after_MB` (target)。

before/after 共 357 个同名文件。分为两类：

| 类型 | 文件数 | 说明 |
|------|--------|------|
| **同帧数、有修改** | ~181 | 修改了 pose（keypose 修正），帧数不变。可直接配对 |
| **不同帧数** | ~176 | 可能剪辑/重制了动作。不用于本次测评 |

### 1.2 筛选标准

从 181 个同帧数文件中，按以下规则筛选：

1. **帧数 ≥ 30**：过短动作无法体现修复效果
2. **帧数 ≤ 360**：超出模型支持长度
3. **最大帧级 pose diff > 0.05 rad**：确保有实质性的 keypose 修改（而非噪声级别差异）
4. **至少 3 帧有显著差异 (per-frame mean diff > 0.1)**：确保修改不是单帧抖动

预估可用测评数据：约 100-150 条。

### 1.3 Keypose 提取规则

从 target motion 中提取 keyposes：

```
1. 计算每帧的 src-target pose 差异: diff[t] = |before[t].poses - after[t].poses|.mean()
2. 选择差异最大的 K 帧作为 keypose（K = max(1, T // 30)，即每 30 帧至少 1 个 keypose）
3. 保证 keypose 间距 ≥ 10 帧（避免过密）
4. 额外保留首帧和尾帧作为 anchor（mask=0）
```

最终得到每条测评数据的 triplet：
- **src_motion**: before 版本（需修正的动作）
- **keyposes**: 从 target 中提取的关键帧（帧索引 + 帧数据）
- **target_motion**: after 版本（Ground Truth）

## 2. Imputation 方案细节

### 2.1 Mask 构建策略

对于 HyMotion M2M（mask-aware noise 变体）：

```
mask = ones(T, 135)      # 初始全部 mask（需重新生成）
mask[0, :] = 0            # 保留首帧
mask[-1, :] = 0           # 保留尾帧
mask[keypose_indices, :] = 0  # 保留 keypose 帧
```

这对应 M6（keyframe_sparse）训练策略，模型已学过此 pattern。

### 2.2 非 keypose 帧的处理

**方案选择**：非 keypose 帧使用 **全 mask（值=1）**，由模型完全重新生成。

理由：
1. mask-aware noise（_man）训练时，mask=1 区域从 noise 生成，mask=0 区域在 x_t 中保持 clean。推理时通过 replacement guidance 保持已知帧精确。
2. 如果对非 keypose 帧加噪后去噪（partial denoise），需要额外设计噪声强度，且 mask-aware noise 训练不支持连续的噪声 level。
3. 全 mask 模式下，首帧/尾帧/keypose 帧通过 VACE inactive 通道提供条件，模型自然会在这些锚点之间做插值过渡。

### 2.3 Replacement Guidance 配置

| 模型 | replacement_guidance | 说明 |
|------|---------------------|------|
| HyMotion M2M `*_man` | `skip_last` | 每步替换已知帧为 clean values，最后一步不替换 |
| MoGenDIT | N/A（内置 imputation） | MoGenDIT 天然支持 mask-aware imputation |

### 2.4 MoGenDIT Imputation

MoGenDIT 使用 DDPM + mask-aware noise，天然支持 imputation。对于 keypose guidance：
1. 构造 (T, 201) motion，在 keypose 帧填入 target pose
2. 构造 obs_mask (T, 201)，keypose 帧 + 首尾帧为 1（已知）
3. 调用 refiner.refine() 的 denoise 模式

注意：MoGenDIT 使用 201-dim 表示（pose_r6d + joint_pos + trans），需要格式转换。

## 3. 测评模型清单

| 模型 ID | Config | 训练方法 | rotation | Checkpoint |
|---------|--------|----------|----------|------------|
| `uncond_fm_man` | hymotion_m2m_completion_uncond_fm_man_046b | FM + MAN | local | epoch_407 |
| `uncond_fm_man_globalrot` | ...uncond_fm_man_globalrot_046b | FM + MAN | global | epoch_88 |
| `uncond_jit_man` | ...uncond_jit_man_046b | JiT + MAN | local | epoch_374 |
| `uncond_jit_man_globalrot` | ...uncond_jit_man_globalrot_046b | JiT + MAN | global | epoch_79 |
| `mogendit` | MoGenDIT MoreDiff-0.1B | DDPM imputation | global | latest |

## 4. 评测指标

| 指标 | 说明 |
|------|------|
| **Keypose MPJPE** | keypose 帧处 output vs target 的 L2 误差（rot6d 空间） |
| **Overall MPJPE** | 全帧 output vs target 的平均误差 |
| **Anchor Preservation** | 首帧/尾帧/keypose 帧的保持精度 |
| **Smoothness** | 相邻帧差异的标准差（越低越平滑） |
| **质量检查通过率** | 修复后动作通过 MotionQualityChecker 的比例 |

## 5. 可视化网站

参考 `motion_annot_web/m2m_repair_compare`，构建 Flask 可视化应用，每条 case 展示：
1. **Source motion** (before)
2. **Keypose 标注**（标记哪些帧是 keypose，用颜色高亮）
3. **各模型输出**（5 个模型 × 1 个结果）
4. **Ground Truth** (after/target)

端口：8095
