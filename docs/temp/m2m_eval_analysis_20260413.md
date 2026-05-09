# M2M Evaluation Analysis (2026-04-13)

## Summary Dashboard

![Summary Dashboard](fig_summary_dashboard.png)

## 评测配置

- 11 模型 × 12 任务 × 100 cases = 13,200 evaluation cases
- 50 inference steps, eval data from `data/eval/hymotion_m2m/`
- Quality check: MotionQualityChecker (15 checkers) on all outputs

## 模型总览

| Model | Architecture | Depth | Epoch | 参数规模 |
|-------|-------------|-------|-------|---------|
| uncond_fm_man | HunyuanMotionMMDiT | 18 | 1000 | 0.46B |
| uncond_fm_man_globalrot | HunyuanMotionMMDiT | 18 | 527 | 0.46B |
| dit_fm_man_s | HunyuanMotionDiT | 12 | 908 | ~49M |
| dit_fm_man_b | HunyuanMotionDiT | 18 | 1000 | ~288M |
| dit_fm_man_l | HunyuanMotionDiT | 24 | 362 | ~383M |
| dit_fm_man_globalrot_s | HunyuanMotionDiT | 12 | 901 | ~49M |
| dit_fm_man_globalrot_b | HunyuanMotionDiT | 18 | 874 | ~288M |
| dit_fm_man_globalrot_l | HunyuanMotionDiT | 24 | 376 | ~383M |
| dit_noinact_fm_man_s | HunyuanMotionDiT | 12 | 1000 | ~49M |
| dit_noinact_fm_man_b | HunyuanMotionDiT | 18 | 704 | ~288M |
| dit_noinact_fm_man_l | HunyuanMotionDiT | 24 | 528 | ~383M |

---

## 指标体系与可靠度

### 指标分层

| 层级 | 指标 | 计算空间 | Global/Local 偏差 | 可靠度 |
|------|------|---------|------------------|--------|
| **L1: 位移精度 (Translation)** | `trans_err_mm`, `masked_trans_err_mm`, `kf_trans_err_mm`, `traj_ade_mm`, `traj_fde_mm` | 绝对坐标系 | 无偏差 | ⭐⭐⭐ 高 |
| **L2: 旋转精度 (Rotation)** | `rot_err`, `masked_rot_err`, `masked_joint_rot_err`, `kf_rot_err` | local joint 的 rot6d 空间 | ⚠️ 有偏差 | ⭐⭐ 中 |
| **L3: 平滑度 (Smoothness)** | `jitter`, `boundary_jerk` | mixed (含 rot6d dims) | ⚠️ 有偏差 | ⭐⭐ 中 |
| **L4: 物理合理性 (Quality Check)** | `quality_pass_rate` | 3D 笛卡尔空间 (FK joint positions) | 无偏差 | ⭐⭐⭐ 高 |
| **L5: 任务特定 (Task-Specific)** | `loop_cont_err`, `traj_ade_mm`, `traj_fde_mm` | 绝对坐标系 | 无偏差 | ⭐⭐ 中 |

### 详细说明

**L1: 位移精度 (Translation)**
- 计算方式：绝对位移空间的误差，单位 mm。计算方式：pred[:, :3] vs gt[:, :3] 的 L2 距离
- Global/Local 偏差：无偏差 — 位移在绝对坐标系计算，与旋转表示无关
- 可靠度说明：直接度量空间位置准确性，物理意义明确

**L2: 旋转精度 (Rotation)**
- 计算方式：rot6d 空间的 L2 距离。计算方式：pred[:, 3:135] vs gt[:, 3:135]，reshape 为 (T, 22, 6) 后 per-joint L2
- Global/Local 偏差：⚠️ 有偏差 — global rotation 模型的 rot6d 是 parent-relative（不含全局旋转），而 local rotation 模型的 rot6d 直接是 local joint rotation。相同物理姿态在两种表示下 rot6d 值不同，L2 距离不可直接比较
- 可靠度说明：rot6d L2 距离不等于角度误差，且受旋转表示影响。同一对模型内对比有效，跨 global/local 对比需谨慎

**L3: 平滑度 (Smoothness)**
- 计算方式：jitter = 全序列加速度幅度均值 (pred[2:]-2*pred[1:-1]+pred[:-2])；boundary_jerk = mask 边界处的 jerk
- Global/Local 偏差：⚠️ 有偏差 — jitter 在 135-dim 空间（含 rot6d）计算，global rotation 表示下 rot6d 的变化模式不同，导致 jitter 被系统性高估
- 可靠度说明：jitter 混合了位移和旋转维度，无量纲。boundary_jerk 仅在有 mask 边界的任务有意义

**L4: 物理合理性 (Quality Check)**
- 计算方式：MotionQualityChecker: 15 个 checker 组成，包括 foot_sliding, jitter, joint_jump, candy_wrapper, arm_penetration, joint_twist, rotation_velocity 等。基于 FK 后的 3D joint positions 和物理约束
- Global/Local 偏差：无偏差 — 所有 checker 基于 FK 后的 3D joint positions，与输入旋转表示无关
- 可靠度说明：综合性物理质量指标，不受旋转表示影响，但是 pass/fail 二值化丢失了连续信息

**L5: 任务特定 (Task-Specific)**
- 计算方式：loop_cont_err = T4 首末帧差异; traj_ade/fde = T8 轨迹误差
- Global/Local 偏差：无偏差
- 可靠度说明：仅在特定任务有数据，样本量小

### 跨 Global/Local 对比时的指标选择

| 可用于跨表示对比 | 不可直接用于跨表示对比 |
|-----------------|---------------------|
| trans_err_mm, masked_trans_err_mm | rot_err, masked_rot_err, masked_joint_rot_err |
| kf_trans_err_mm, traj_ade/fde_mm | kf_rot_err |
| quality_pass_rate | jitter (混合 rot6d dims) |
| loop_cont_err (绝对位移) | boundary_jerk (混合 rot6d dims) |

---

## 1. Global Rotation vs Local Rotation

![Global vs Local — Wins by Metric Layer](fig_global_vs_local.png)

*斜线填充 = 有度量偏差的指标层（L2 Rotation, L3 Smoothness），跨 global/local 对比时不可靠。*

![Global vs Local — Per-Task Heatmap (unbiased metrics)](fig_global_local_heatmap.png)

### 1.1 Epoch 公平性

| 对比组 | Local Epoch | Global Epoch | 差距 | 可信度 |
|--------|------------|-------------|------|--------|
| DiT-S | 908 | 901 | <1% | ✅ 高 |
| DiT-L | 362 | 376 | 4% | ✅ 高 |
| DiT-B | 1000 | 874 | 14% | ⚠️ 中 |
| MMDiT | 1000 | 527 | 90% | ❌ 不可信 |

### 1.2 DiT-S 按指标层级对比（最公平：908 vs 901 epoch）

| 指标层级 | Local wins | Global wins | ~same | 趋势 |
|---------|------------|------------|--------|------|
| L1 | 15 | 13 | 2 | 接近 |
| L2 | 30 | 3 | 7 | **Local** |
| L3 | 16 | 3 | 1 | **Local** |
| L4 | 3 | 9 | 0 | **Global** |
| L5 | 1 | 2 | 0 | 接近 |
| **TOTAL** | **65** | **30** | **10** | **Local** |

### 1.3 DiT-L 按指标层级对比（362 vs 376 epoch）

| 指标层级 | Local wins | Global wins | ~same | 趋势 |
|---------|------------|------------|--------|------|
| L1 | 15 | 11 | 4 | **Local** |
| L2 | 36 | 1 | 3 | **Local** |
| L3 | 17 | 2 | 1 | **Local** |
| L4 | 3 | 9 | 0 | **Global** |
| L5 | 2 | 1 | 0 | 接近 |
| **TOTAL** | **73** | **24** | **8** | **Local** |

### 1.4 仅看无偏差指标（trans_err 系列 + quality%）

**DiT-S (908 vs 901ep)**
- Trans 系列: Local 15, Global 13, ~same 2
- Quality%: Local 3, Global 9, ~same 0
- **无偏差指标合计: Local 18, Global 22, ~same 2**

**DiT-L (362 vs 376ep)**
- Trans 系列: Local 15, Global 11, ~same 4
- Quality%: Local 3, Global 9, ~same 0
- **无偏差指标合计: Local 18, Global 20, ~same 4**

### 1.5 Global vs Local 结论

| 结论维度 | 发现 |
|---------|------|
| 全量指标 (含有偏差) | Local 大幅领先 (DiT-S: 64:29, DiT-L: 72:23)，但 rot_err/jitter 系列贡献了绝大多数 Local wins |
| 仅无偏差指标 (trans + quality) | 接近或 Global 略优 — quality% Global 大幅领先 (两组一致: 3:9) |
| T4 Completion | **Global 唯一强势任务**：DiT-S trans_err -17%，所有指标 Global 全赢 |
| T7 Repair | Global trans_err 大幅好 (-18%~-21%)，但 quality% 反而更低 |

> **核心结论**：由于 rot_err 和 jitter 系列存在系统性度量偏差（占 Local wins 的 ~70%），
> 不能直接得出 "Local 更好" 的结论。看无偏差指标（trans_err + quality%），
> **两者整体差异不大**，Global 在物理质量 (quality%) 上略优，Local 在位移精度上略优。
> Global rotation 在 T4 (completion) 上有显著优势。

---

## 2. 模型规模 (DiT S / B / L)

![Model Size S vs B — Wins by Metric Layer](fig_model_size.png)

### 2.1 Epoch 差异

| 规模 | Local Epoch | Globalrot Epoch | Noinact Epoch |
|------|-----------|----------------|--------------|
| S | 908 | 901 | 1000 |
| B | 1000 | 874 | 704 |
| L | 362 | 376 | 528 |

> ⚠️ L 模型 epoch 远低于 S/B，**所有 L 的劣势可能是训练不足而非规模不行**。

### 2.2 S vs B 按指标层级（最佳对比：Globalrot 901 vs 874ep）

| 指标层级 | S wins | B wins | ~same | 趋势 |
|---------|------------|------------|--------|------|
| L1 | 7 | 21 | 2 | **B** |
| L2 | 10 | 27 | 3 | **B** |
| L3 | 2 | 16 | 2 | **B** |
| L4 | 8 | 3 | 1 | **S** |
| L5 | 2 | 1 | 0 | 接近 |
| **TOTAL** | **29** | **68** | **8** | **B** |

### 2.3 S vs B Local（908 vs 1000ep，B 多训 10%）

| 指标层级 | S wins | B wins | ~same | 趋势 |
|---------|------------|------------|--------|------|
| L1 | 3 | 25 | 2 | **B** |
| L2 | 13 | 24 | 3 | **B** |
| L3 | 5 | 14 | 1 | **B** |
| L4 | 4 | 7 | 1 | **B** |
| L5 | 1 | 2 | 0 | 接近 |
| **TOTAL** | **26** | **72** | **7** | **B** |

### 2.4 规模结论

- **DiT-B 全面优于 DiT-S**：两组对比一致 (Globalrot 67:28, Local 70:26)，覆盖所有指标层级
- Globalrot S vs B 是最公平对比（901 vs 874ep），B 仍全面赢 → 规模优势是真实的
- 但 **Globalrot S 的 quality% 反而更好** (8:3)，说明 S 模型生成的动作物理合理性更高
- DiT-L 因训练不足无法下结论，需继续训练到 800+ ep

---

## 3. Inactive 通道的影响

![Inactive Channel — Wins by Metric Layer](fig_inactive.png)

### 3.1 Epoch 公平性

| 对比组 | w/ inactive | no inactive | 差距 | 可信度 |
|--------|-----------|-----------|------|--------|
| DiT-S | 908 ep | 1000 ep | noinact 多训 10% | ✅ 较高 |
| DiT-B | 1000 ep | 704 ep | inactive 多训 42% | ❌ 不可信 |
| DiT-L | 362 ep | 528 ep | noinact 多训 46% | ❌ 不可信 |

### 3.2 DiT-S 按指标层级对比（908 vs 1000ep）

| 指标层级 | w/inact wins | noinact wins | ~same | 趋势 |
|---------|------------|------------|--------|------|
| L1 | 15 | 11 | 4 | **w/inact** |
| L2 | 14 | 20 | 6 | **noinact** |
| L3 | 10 | 10 | 0 | 接近 |
| L4 | 9 | 2 | 1 | **w/inact** |
| L5 | 3 | 0 | 0 | **w/inact** |
| **TOTAL** | **51** | **43** | **11** | **w/inact** |

### 3.3 Inactive 通道结论

- 全量指标 **49:43 接近持平**
- L1 位移精度：持平 (11:13)
- L2 旋转精度：noinact 略好 (13:21)，但 noinact 多训了 10%
- L3 平滑度：方向相反 — jitter noinact 好 (3:9)，boundary_jerk w/inact 好 (7:1)
- L4 物理质量：**w/inactive 大幅领先** (9:2)

> **核心结论**：删除 inactive 通道对位移/旋转精度**影响很小**。
> 但 w/inactive 的**物理质量 (quality%) 显著更好** (9:2)，且 noinact 多训了 10%。
> 如果修正训练量差异，w/inactive 可能在 quality 上优势更大。
> 建议**保留 inactive 通道**。

---

## 4. 综合排名

### 4.1 各任务 Top-3（Primary Metric）

| Task | Name | 🥇 1st | 🥈 2nd | 🥉 3rd |
|------|------|--------|--------|--------|
| T1 | In-Between | dit_noinact_fm_man_s (1000ep) | dit_fm_man_s (908ep) | uncond_fm_man (1000ep) |
| T2 | KF Interp | dit_fm_man_globalrot_b (874ep) | uncond_fm_man (1000ep) | dit_fm_man_b (1000ep) |
| T3 | Future Pred | dit_noinact_fm_man_s (1000ep) | dit_fm_man_b (1000ep) | dit_fm_man_s (908ep) |
| T4 | Completion | dit_fm_man_globalrot_s (901ep) | uncond_fm_man_globalrot (527ep) | dit_fm_man_globalrot_b (874ep) |
| T5 | Continuation | dit_fm_man_globalrot_b (874ep) | dit_fm_man_b (1000ep) | dit_fm_man_l (362ep) |
| T6 | Joint Compl | uncond_fm_man (1000ep) | dit_noinact_fm_man_l (528ep) | dit_fm_man_s (908ep) |
| T7 | Repair | dit_fm_man_globalrot_l (376ep) | dit_fm_man_globalrot_b (874ep) | uncond_fm_man_globalrot (527ep) |
| T8 | Unconditional | uncond_fm_man (1000ep) | dit_noinact_fm_man_b (704ep) | dit_fm_man_s (908ep) |
| T9 | Up 5fps | dit_noinact_fm_man_b (704ep) | dit_fm_man_b (1000ep) | dit_noinact_fm_man_l (528ep) |
| T10 | Up 1fps | dit_fm_man_b (1000ep) | uncond_fm_man (1000ep) | dit_noinact_fm_man_l (528ep) |
| T11 | Up 0.5fps | uncond_fm_man (1000ep) | uncond_fm_man_globalrot (527ep) | dit_noinact_fm_man_l (528ep) |
| T12 | Up Auto | uncond_fm_man (1000ep) | dit_fm_man_b (1000ep) | uncond_fm_man_globalrot (527ep) |

### 4.2 总 Wins

| Model | Epoch | 🥇 Wins | Top-3 次数 |
|-------|-------|---------|-----------|
| uncond_fm_man | 1000 | 4 | 7 |
| dit_noinact_fm_man_s | 1000 | 2 | 2 |
| dit_fm_man_globalrot_b | 874 | 2 | 4 |
| dit_fm_man_globalrot_s | 901 | 1 | 1 |
| dit_fm_man_globalrot_l | 376 | 1 | 1 |
| dit_noinact_fm_man_b | 704 | 1 | 2 |
| dit_fm_man_b | 1000 | 1 | 6 |

### 4.3 关键观察

- **T4 和 T7 的 Top-3 全部是 globalrot 模型**，确认 global rotation 在 completion/repair 任务上有显著优势
- **dit_fm_man_b Top-3 次数最多 (6次)**，是最稳定的全能模型
- **uncond_fm_man (MMDiT 0.46B) 4 次第一**，大参数量在无条件/极稀疏任务上有优势
- 注意：primary metric 中 T3/T6/T8 用 rot_err 系列，在 global vs local 对比中可能有偏差

---

*Generated: 2026-04-13. Data: eval_results/m2m/summary.json*
*Analysis covers all 13 metrics organized by measurement layer, with reliability assessment for cross-representation comparisons.*