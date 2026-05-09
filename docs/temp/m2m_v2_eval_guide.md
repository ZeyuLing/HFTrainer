# HyMotion M2M v2 评测指南

> **版本**: v1.2 | **日期**: 2026-04-13
> **评测脚本**: `tools/eval_m2m_v2_all_tasks.py`
> **指标模块**: `hftrainer/evaluation/motion/m2m_eval_metrics.py`
> **任务定义**: `hftrainer/evaluation/motion/m2m_eval_tasks.py`

---

## 1. v2 模型概况

### 1.1 v2 核心变化（vs v1）

| 维度 | v1 | v2 |
|------|----|----|
| **Motion dim** | 135（trans + rot6d） | **198**（trans + rot6d + **position**） |
| **Position 通道** | 无 | 21 joints × 3D（XZ 相对 pelvis，Y 绝对） |
| **VACE 模式** | `split_reactive`（4 通道，540-dim） | **`no_inactive`**（3 通道，594-dim） |
| **输入结构** | `[x_t, inactive, reactive, mask]` | `[x_t, reactive, mask]` |
| **Condition sampler** | 单层 7 策略（M1-M7） | **两层架构**（tier2_prob=0.4） |
| **MAN** | 可选 | **默认开启** |
| **训练数据** | 549K 全量（含 85K 低质量） | **456K 高质量**（已过滤） |
| **FK consistency loss** | 无 | **0.1 权重**（warmup 2000 steps） |

### 1.2 198-dim 布局

```
dims [0:3]       — translation（3D 绝对位移）
dims [3:135]     — 22 joints × 6D rot6d（行主序）
dims [135:198]   — 21 joints × 3D position（pelvis 除外）
                    每个关节：[X_rel_pelvis, Y_absolute, Z_rel_pelvis]
```

> **为什么 198-dim？** 增加显式 position 通道使模型可直接约束世界坐标位置（end-effector、轨迹），不再需要 FK/IK 转换。KIMODO 的核心优势（global position imputation）在 v2 中通过 position 通道实现。

### 1.3 待评测模型

| 简称 | 文本条件 | 旋转空间 | Epoch | Work Dir |
|------|---------|---------|------:|---------|
| **uncond_local** | ❌ | Local | 119 | `work_dirs/hymotion_m2m_v2_uncond_local_046b` |
| **uncond_global** | ❌ | Global | 116 | `work_dirs/hymotion_m2m_v2_uncond_global_046b` |
| **caption_local** | ✅ Qwen3+CLIP | Local | 86 | `work_dirs/hymotion_m2m_v2_caption_local_046b` |
| **caption_global** | ✅ Qwen3+CLIP | Global | 91 | `work_dirs/hymotion_m2m_v2_caption_global_046b` |

4 个模型形成 **2×2 消融矩阵**（caption × rotation space），所有模型共享：
- 架构: HunyuanMotionMMDiT 0.46B（18 层，feat_dim=1024）
- 预训练: T2M 1.0-Lite 权重（input/output 层随机初始化）
- 训练数据: `train_hymotion_400h_hq_20260403.json`（456K 高质量）
- MAN + `skip_last` imputation

---

## 2. 评测任务总览（E1-E13）

### 2.1 任务矩阵

| # | 任务 | KIMODO 可比 | 数据文件 | 样本数 | Caption | 核心指标 |
|---|------|:---------:|---------|------:|:-------:|---------|
| **E1** | Text-to-Motion | ✅ | 251125_yiran_subset.json | 240 | ✅ | FID, R-precision, Diversity |
| **E2** | Motion In-Betweening | ✅ | eval_transition.json | 500 | ❌ | MPJPE, 边界平滑度, Jitter |
| **E3** | 稀疏关键帧插值 | ✅ | eval_keyframe.json | 500 | 144 | MPJPE, Jitter, Bone CV |
| **E4** | 末端位置约束 / Text+Keypose | ✅ | eval_transition.json | 500 | ❌/✅ | EE error, MPJPE |
| **E5** | 轨迹跟随 | ✅ | eval_trajectory.json | 500 | 130 | ADE/FDE, Skating |
| **E6** | 脚接地约束（Rotation/Position） | ⚠️ | eval_transition.json | 500 | ❌ | Penetration, Float, Skating |
| **E7** | First-frame 续写 | ❌ | eval_first_frame_cond.json | 300 | ✅ | MPJPE@1, Jitter, FID |
| **E8** | Loop 动画（+轨迹） | ❌ | eval_loop_animation.json | 200 | ✅ | Loop error, Diversity |
| **E9** | 动作修复 | ❌ | eval_repair.json | 973 | ❌ | QPass rate, Defect 减少 |
| **E10** | Part-level 控制 | ❌ | eval_transition.json | 500 | ❌ | 保持区 MPJPE, 生成区质量 |
| **E11** | Caption 条件补全 | ❌ | eval_transition_with_caption.json | 300 | ✅ | R-precision, MPJPE |
| **E12** | Multi-Prompt 自回归生成 | ❌ | eval_first_frame_cond.json | 300 | ✅ | 段间平滑度, Jitter, Skating |

### 2.2 执行优先级

```
Phase 1（核心，直接跟 KIMODO 比）         Phase 2（自比任务）              Phase 3（消融与进阶）
├─ E2 In-Betweening                      ├─ E9 Repair (vs MoGenDIT)       ├─ E12 Local vs Global
├─ E3 Keyframe                           ├─ E7 First-Frame                ├─ E6 Foot Ground
├─ E5 Trajectory                         ├─ E1 T2M (需 FID/R-prec)       ├─ E10 Part-Level
└─ E4 End-Effector                       └─ E13 Multi-Prompt              ├─ E8 Loop
                                                                          └─ E11 Caption Completion
```

---

## 3. 每个任务的详细设定

### E1: Text-to-Motion 生成

**Task**: 给定文本描述，从纯噪声生成完整动作。mask 全 1。

| Setting | 说明 |
|---------|------|
| default | 标准 T2M，CFG scale=7.5 |

**文本来源**: `data/eval/t2m/251125_yiran_subset.json`（240 条人工审核提示，格式 `text#frames#cond#id`）。该文件无对应 GT motion，因此 E1 **不计算 MPJPE**，只评生成质量指标。

**KIMODO 对比设置**:
- KIMODO: `text + separated CFG (w_text=2)` → DDIM 100 steps
- M2M v2: `text + CFG (w=7.5)` → Euler 50 steps，mask=全1，reactive=全0

**Mask**: `build_full_mask(T, 198)` → 全 1

**指标**: FID↓, R-precision (top1/3)↑, Diversity↑, Jitter↓, Bone CV↓, Skating↓

> ⚠️ FID/R-precision 需要特征提取器（TMR / CLIP4Motion），当前脚本**尚未实现**，需后续补充。

---

### E2: Motion In-Betweening（过渡补全）

**Task**: 给定首 N 帧和尾 M 帧（全关节 rotation + position），补全中间部分。

| Setting | 首帧 | 尾帧 | 特殊条件 |
|---------|------|------|---------|
| **A** | 5 帧 | 5 帧 | 标准 |
| **B** | 5 帧 | 5 帧 | 序列 >200 帧（长距离） |
| **C** | 30 帧 | 5 帧 | 前长后短（不对称） |

**Mask**: `build_inbetween_mask(T, D, keep_start=N, keep_end=M)`

**MAN 推理**: `replacement_guidance='skip_last'`
- 每步 ODE 中 keep 区域用 `clean_motion` 替换
- 边界帧精确保持（MPJPE_unmasked ≈ 0）

**指标**:
| 指标 | 说明 | 预期 |
|------|------|------|
| mpjpe_masked↓ | 生成区域 vs GT 关节位置误差 | 核心指标 |
| mpjpe_unmasked↓ | 保持区域误差（MAN → ≈0） | 验证 imputation |
| boundary_accel_jump↓ | mask 边界处加速度跳变 | 过渡质量 |
| jitter_pos↓ | 生成区域抖动（m/s³） | 平滑性 |
| bone_length_cv_mean↓ | 骨骼长度变异系数 | 结构一致 |
| foot_skating_ratio↓ | 脚滑帧比例 | 物理合理 |

---

### E3: 稀疏关键帧插值

**Task**: 每隔 K 帧给一个完整 full-body 关键帧，插值中间帧。

| Setting | 间隔 | 说明 |
|---------|------|------|
| **A** | 30 帧 | 1s@30fps，标准 |
| **B** | 60 帧 | 2s，较稀疏 |
| **C** | 15 帧 | 0.5s，较密 |
| **D** | 10-90 帧 | 随机间距（非均匀） |

**Mask**: `build_keyframe_mask(T, D, interval=K)` — 关键帧 mask=0，其余 mask=1

**指标**: 同 E2 + keyframe accuracy（关键帧处精度，MAN → ≈0）

---

### E4: 末端位置约束（End-Effector）/ Text+Keypose

**Task**: 指定部分帧的部分关节 3D 位置，生成满足约束的完整动作。新增 Text+Keypose 模式：用文本+关键帧姿态作为首帧/尾帧约束。

| Setting | 约束关节 | 间隔 | 说明 |
|---------|---------|------|------|
| **A** | 右手（r_wrist） | 每 10 帧 | 经典 end-effector |
| **B** | 双脚踝（l_ankle, r_ankle） | 每 15 帧 | 脚踝位置约束 |
| **C** | 右手+左脚（r_wrist, l_foot） | 每 15 帧 | 多关节组合 |
| **D** | Text + keypose P（首帧） | — | 文本引导从 P 开始生成 |
| **E** | Text + keypose P（首+尾帧） | — | 文本引导在两个 keypose 间生成 |

**Setting D/E（Text+Keypose）**:
- 从 eval 数据中挑选一个关键帧姿态 P
- D: P 作为首帧约束，文本引导模型从 P 开始生成后续动作
- E: P 作为首帧和尾帧约束，文本引导模型生成 P→P 之间的动作
- 这测试模型同时满足**空间约束**（keypose）和**语义约束**（文本）的能力
- Mask: `build_text_keypose_mask(T, D, keep_start=1, keep_end=0/1)`

**Mask (A-C)**: `build_end_effector_mask(T, D, joint_names, frame_interval)`
- 约束帧的指定关节 group → mask=0
- 其余全部 mask=1

**v2 优势**: 198-dim 包含 position 通道，可直接在 position dims 上约束，无需 IK。这是 **v2 相对 v1 的核心新能力**。

**约束来源**: 从 GT motion 的 FK 结果中提取指定关节的世界坐标位置。

**指标**:
| 指标 | 说明 |
|------|------|
| ee_error_mean↓ | 约束帧处 FK 输出 vs 约束位置 L2 距离 |
| ee_error_max↓ | 最大约束偏差 |
| mpjpe_masked↓ | 生成区域整体误差 |
| jitter_pos↓ | 自然度 |

---

### E5: 轨迹跟随

**Task**: 给定 root 的 XZ 平面轨迹（和可选 heading），生成跟随轨迹的全身动作。

| Setting | 模式 | 说明 |
|---------|------|------|
| **A** | dense | 每帧给 root XZ 位置 |
| **B** | sparse | 每 30 帧给 root XZ（途径点） |
| **C** | trajectory_heading | 每帧 XZ + pelvis 朝向 |
| **D** | heading_only | 每 30 帧给 heading，无位置约束 |

**Mask**: `build_trajectory_mask(T, D, mode, interval, include_heading)`
- `dense`: translation group 全帧 mask=0
- `sparse`: 每 interval 帧 translation group mask=0
- `heading_only`: pelvis rotation group mask=0

**指标**: Trajectory ADE↓, FDE↓, Heading error↓, Skating↓, Jitter↓

---

### E6: 脚接地约束

**Task**: 在检测到的接地帧，约束脚踝使其贴地。区分两种根本不同的约束方式：

#### Rotation 约束（135-dim joint group 级别）

约束脚踝的旋转角度（rot6d），使脚踝朝向与 GT 接地一致。这是 **间接约束**——通过 FK 链传递才影响世界位置。

| Setting | 说明 |
|---------|------|
| **A_rot** | 从 GT 检测接地帧 → 约束 ankle rotation joint group |
| **B_rot** | 全帧约束 ankle rotation joint group |

#### Position 约束（198-dim per-dim 级别）

直接约束脚踝的世界坐标位置通道（dims [135:198]），是 **直接约束**。v2 的 position 通道使这种细粒度约束成为可能。

| Setting | 约束轴 | 说明 |
|---------|--------|------|
| **C_pos_y** | 仅 Y 轴 | 约束脚踝高度=0（最小侵入，XZ 自由移动） |
| **D_pos_xz** | 仅 XZ 轴 | 水平位置锁定（防止脚滑），Y 自由 |
| **E_pos_xyz** | 全 XYZ | 完整 3D 位置约束（最强但可能过约束） |

> **设计理念**：C_pos_y 是最自然的接地约束——只要求"脚不穿地面"（Y≥0），不限制水平运动。D_pos_xz 针对脚滑问题——接地时水平不动。E_pos_xyz 是完整约束，适合验证模型的约束遵从能力。

**接地帧检测**: 从 GT FK positions 中检测 ankle/foot Y < 5cm 的帧。

**指标**: Penetration↓, Float↓, Skating ratio↓, Jitter↓

---

### E7: First-Frame 续写

**Task**: 给定第 1 帧完整 pose + text caption，生成后续动作。

**Mask**: `build_first_frame_mask(T, D)` → frame 0 全 0，其余全 1

**指标**: mpjpe_unmasked↓（首帧还原 ≈0），Jitter↓，Skating↓

> 只跟自己比（v1 / HY-Motion 1.0），KIMODO 无直接对标。

---

### E8: Loop 动画

**Task**: 生成首尾帧一致的循环动画，给定 text caption。支持额外的轨迹约束。

| Setting | 模式 | 说明 |
|---------|------|------|
| **A** | 经典 loop | 仅首=尾帧约束 |
| **B** | Loop + 密集轨迹 | 首=尾帧 + 每帧 root XZ 轨迹约束 |
| **C** | Loop + 稀疏途径点 | 首=尾帧 + 每 30 帧 root XZ 途径点 |

**Mask**:
- A: `build_loop_mask(T, D, trajectory_mode='none')` → frame 0 和 frame T-1 均 mask=0
- B: frame 0/T-1 全 mask=0 + 所有帧 translation group mask=0
- C: frame 0/T-1 全 mask=0 + 途径帧 translation group mask=0

**设计理念**:
- **A（经典 loop）**: 最基本的循环能力测试——首尾一致即可
- **B（密集轨迹 loop）**: 给定一条闭合轨迹（如圆形/8 字形），要求角色沿轨迹走一圈并回到起点，测试轨迹跟随+循环一致性
- **C（稀疏途径点 loop）**: 只给几个途径点，模型自行规划路径并确保循环

**MAN 推理**: 首尾帧从同一 GT 帧注入 → 首尾一致。

**指标**:
| 指标 | 说明 |
|------|------|
| loop_position_error↓ | 首尾帧 MPJPE |
| loop_velocity_error↓ | 首尾帧速度差 |
| trajectory_ade↓ | 轨迹误差（B/C only） |
| jitter_pos↓ | 抖动 |

---

### E9: 动作修复

**Task**: 给定有缺陷的动作，修复为高质量动作。reactive=corrupted motion。

| Setting | 说明 |
|---------|------|
| **A** | 用 quality checker 自动检测 → 膨胀 mask → 修复 |
| **B** | Oracle mask（GT 缺陷区域） |
| **C** | 全 mask 重建（最保守） |

**模式**: **Editing 模式**（`is_editing=True`）— src_motion **不清零** mask=1 区域，reactive 传入 LQ motion。

**对比**: MoGenDIT `ada_denoise` + M2M v1

**测试数据分布**: 973 条按缺陷类型分组
| 类型 | 数量 | 说明 |
|------|------|------|
| foot_sliding | 631 | 最高频 |
| candy_wrapper | 111 | 糖纸旋转 |
| jitter | 90 | 抖动 |
| joint_jump | 87 | 关节跳变 |
| rotation_velocity | 76 | 旋转速度异常 |
| 其他 | 119 | neck/ankle/arm_penetration |

**指标**: QPass rate↑, Defect count 变化↓, Jitter↓, Skating↓

---

### E10: Part-Level 控制

**Task**: 固定部分关节，只重新生成其他关节。

| Setting | 保持区域 | 生成区域 |
|---------|---------|---------|
| **A** | 上身（spine/arms/head）+ pelvis | 下肢（hips/knees/ankles/feet） |
| **B** | 下肢 + translation + pelvis | 上身 |
| **C** | 仅 translation + pelvis rotation | 全身 pose |

**指标**: 保持区 MPJPE↓（应 ≈0），生成区 Jitter↓，Bone CV↓

---

### E11: Caption-conditioned Completion

**Task**: 在 E2/E3 的基础上加入 text caption 作为额外条件。

| Setting | 基础任务 | 说明 |
|---------|---------|------|
| inbetween | E2-A（5+5帧） | In-betweening + caption |
| keyframe | E3-A（30帧间隔） | Keyframe + caption |

**意义**: 对比有/无 caption 模型，验证 caption 是否提升语义一致性。

**指标**: MPJPE↓, Jitter↓, Bone CV↓

---

### E12: Local vs Global Rotation 消融

**Task**: 在 E2-E6 核心任务上对比 4 个模型变体的表现差异。

**关键假设验证**:

| 假设 | 验证任务 | 预期结果 |
|------|---------|---------|
| Global rotation 在位置约束任务上更优 | E4, E5, E6 | Global 的 EE error / ADE 更低 |
| Caption 在语义任务上显著提升 | E1, E7, E11 | Caption 模型 R-precision 更高 |
| Local rotation 在插值任务上更稳定 | E2, E3 | Local 的 MPJPE 可能更低 |

---

### E13: Multi-Prompt 自回归生成

**Task**: 给定 N 条动作文本描述，自回归地生成无限长的连续动作。每个片段由一条文本引导生成，片段间通过 overlap 帧实现平滑过渡。

**生成流程**:

```
Prompt 1: "A person walks forward."     → 生成 Segment 1 (T1 帧，纯生成)
Prompt 2: "The person turns left."      → 取 Seg1 尾部 K 帧作为 Seg2 首部已知帧 → 生成 Seg2
Prompt 3: "The person starts running."  → 取 Seg2 尾部 K 帧作为 Seg3 首部已知帧 → 生成 Seg3
...
最终动作 = concat(Seg1, Seg2[K:], Seg3[K:], ...)
```

| Setting | 文本段数 | Overlap 帧数 | 预期总帧数 |
|---------|---------|:----------:|:---------:|
| **A** | 3 | 5 | ~360 帧（12s） |
| **B** | 5 | 5 | ~600 帧（20s） |
| **C** | 10 | 10 | ~1200 帧（40s） |

**Mask**:
- Segment 1: `build_full_mask(T, D)` — 纯生成
- Segment 2+: `build_multi_prompt_mask(T, D, overlap_frames=K)` — 前 K 帧 keep，其余 generate

**关键设计**:
- **Overlap 区域**：前一段的尾部 K 帧作为下一段的首部已知帧，通过 MAN imputation 保证过渡平滑
- **文本切换**：每段使用不同的 text caption，模型需在保持运动连续性的同时切换语义
- **无限长生成**：理论上可以无限链接，实测关注长序列（>30s）后的质量退化

**指标**:
| 指标 | 说明 |
|------|------|
| segment_boundary_smoothness↓ | 段间边界的加速度跳变 |
| jitter_pos↓ | 全局抖动 |
| bone_length_cv_mean↓ | 全局骨骼一致性 |
| foot_skating_ratio↓ | 全局脚滑 |
| total_duration | 生成总时长（秒） |
| per_segment_r_precision↑ | 每段动作与对应文本的匹配度 |

> **对比 KIMODO**：KIMODO 支持 multi-prompt sequential generation（用 inpainting 在段间过渡），但 M2M v2 的自回归 overlap 方式更自然——不需要显式的 transition 区域规划。

---

## 4. 指标体系

### 4.1 已实现指标

| 类别 | 指标 | 函数 | 依赖 FK | 说明 |
|------|------|------|:------:|------|
| **位置精度** | mpjpe_all | `compute_mpjpe()` | ✅ | 全帧全关节平均位置误差 |
| | mpjpe_masked | `compute_mpjpe(mask=mask)` | ✅ | 仅生成区域 |
| | mpjpe_unmasked | `compute_mpjpe(mask=1-mask)` | ✅ | 仅保持区域（验证 imputation） |
| **时间平滑** | jitter_pos | `compute_jitter_positions()` | ✅ | 关节位置三阶导数 (m/s³) |
| | jitter_135 | `compute_jitter_135()` | ❌ | 135-dim 原始空间抖动 |
| **骨骼一致** | bone_length_cv_mean | `compute_bone_length_cv()` | ✅ | 骨骼长度变异系数均值 |
| | bone_length_cv_max | 同上 | ✅ | 变异系数最大值 |
| **轨迹** | trajectory_ade | `compute_trajectory_metrics()` | ❌ | Root XZ 平均位移误差 |
| | trajectory_fde | 同上 | ❌ | 终点位移误差 |
| | heading_error | `compute_heading_error()` | ✅ | Pelvis 朝向角度误差 (deg) |
| **边界** | boundary_accel_jump | `compute_boundary_smoothness()` | 可选 | mask 边界加速度跳变 |
| **末端约束** | ee_error_mean | `compute_end_effector_error()` | ✅ | 约束点位置误差均值 |
| | ee_error_max | 同上 | ✅ | 最大约束偏差 |
| **循环** | loop_position_error | `compute_loop_continuity()` | ✅ | 首尾帧关节位置差 |
| | loop_velocity_error | 同上 | ✅ | 首尾帧速度差 |
| **脚地面** | foot_penetration | `compute_foot_ground_metrics()` | ✅ | 脚穿地面深度 |
| | foot_float | 同上 | ✅ | 接地帧离地高度 |
| | foot_skating_ratio | 同上 | ✅ | 脚滑帧比例 |
| | foot_avg_skate | 同上 | ✅ | 平均滑动速度 |
| **FK 一致** | fk_consistency | `compute_fk_consistency()` | ✅ | rotation FK vs position 通道差异（198-dim only） |

### 4.2 待实现指标

| 指标 | 依赖 | 说明 | 优先级 |
|------|------|------|--------|
| **FID** | TMR / T2M-GPT 特征提取器 | 生成分布 vs GT 分布距离 | E1 必需 |
| **R-precision** | TMR / CLIP4Motion | 文本-动作匹配度 | E1/E11 必需 |
| **Diversity** | feature space inter-sample L2 | 生成多样性 | E1 |
| **QPass rate** | 16 个 quality checkers | 通过质量检测比例 | E9 |
| **Defect count** | quality checkers | 修复前后缺陷数变化 | E9 |

---

## 5. 使用方法

### 5.1 基本命令

```bash
# Phase 1: 核心任务（E2/E3/E5），每模型 50 样本，快速验证
python tools/eval_m2m_v2_all_tasks.py \
    --tasks E2 E3 E5 \
    --max-samples 50

# Phase 1 完整跑（所有 setting，100 样本）
python tools/eval_m2m_v2_all_tasks.py \
    --tasks E2 E3 E4 E5 \
    --settings A B C \
    --max-samples 100

# Phase 2: 自比任务
python tools/eval_m2m_v2_all_tasks.py \
    --tasks E7 E9 \
    --max-samples 200

# 全量评测
python tools/eval_m2m_v2_all_tasks.py \
    --all-tasks \
    --max-samples 200

# 对比 v1 基线
python tools/eval_m2m_v2_all_tasks.py \
    --models uncond_local v1_uncond_fm_man \
    --tasks E2 E3 E5

# 只跑 caption 模型的文本任务
python tools/eval_m2m_v2_all_tasks.py \
    --models caption_local caption_global \
    --tasks E1 E7 E8 E11 \
    --text-guidance-scale 7.5
```

### 5.2 关键参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--models` | 4 个 v2 模型 | 可选 v1 基线 |
| `--tasks` | E2 E3 E5 | 任务 ID（E1-E12） |
| `--settings` | 每任务全部 | 子设定（A/B/C/D/default） |
| `--max-samples` | 50 | 每任务最大样本数 |
| `--num-steps` | 50 | ODE 积分步数 |
| `--replacement-guidance` | skip_last | MAN imputation 模式 |
| `--text-guidance-scale` | 7.5 | CFG scale（仅 caption 模型生效） |
| `--output-dir` | work_dirs/m2m_v2_eval_report | 结果输出目录 |

### 5.3 输出格式

结果保存为 JSON：`{output_dir}/eval_v2_{timestamp}.json`

```json
{
  "uncond_local": {
    "checkpoint": "work_dirs/.../checkpoint-epoch_119",
    "rotation_space": "local",
    "motion_dim": 198,
    "tasks": {
      "E2_A": {
        "task_id": "E2",
        "setting": "A",
        "num_samples": 50,
        "aggregated": {
          "mpjpe_masked": {"mean": 0.045, "std": 0.012, "median": 0.042, ...},
          "jitter_pos": {"mean": 12.3, "std": 5.1, ...},
          ...
        },
        "per_sample": [...]
      }
    }
  }
}
```

---

## 6. KIMODO 对比方案

### 6.1 可直接对比的任务

| 任务 | KIMODO 方法 | M2M v2 方法 | 差异要点 |
|------|-----------|-----------|---------|
| E2 In-Between | FullBodyConstraintSet → imputation（position+rotation） | mask[0:N]=0, mask[-M:]=0（198-dim），MAN imputation | M2M v2 通过 rotation+position 通道同时约束 |
| E3 Keyframe | FullBodyConstraintSet at keyframe indices | mask[kf_indices]=0（198-dim 全维度） | 同上 |
| E4 End-Effector | EndEffectorConstraintSet → position imputation | mask 指定关节 position dims=0 | v2 新增能力，与 KIMODO 可比 |
| E5 Trajectory | Root2DConstraintSet → smooth_root_pos imputation | mask translation group=0 | 表示方式不同（KIMODO: smooth_root_pos，v2: abs_trans） |

### 6.2 不可直接对比

| 任务 | 原因 | 基线替代 |
|------|------|---------|
| E1 T2M | 不同文本编码器、不同生成步骤 | 在公开 benchmark（HumanML3D test set）上独立评测后对比数值 |
| E6 脚接地 | KIMODO 有显式 foot contact 4-dim | 只比脚接地物理指标，不比 mask/condition 方式 |
| E7-E12 | KIMODO 不支持 | 只跟 v1 / MoGenDIT 比 |

---

## 7. 测试数据来源与构建

### 7.1 路径约定

| 路径 | 说明 |
|------|------|
| `data/eval/hymotion_m2m/*.json` | 评测 datalist JSON（索引文件） |
| `data/hymotion_data/` | 动作 NPZ 文件根目录（symlink → HunyuanMotion） |

> **⚠️ 注意**：eval JSON 中的 `motion_path` 相对于 `data/hymotion_data/`，**不是** `data/motionhub/`。
> 实际加载时通过 `resolve_motion_path()` 在 `[data/hymotion_data/3D/20251111/motions/, data/hymotion_data/]` 两个 root 下查找。
> 参考：`scripts/eval_m2m_all_tasks.py` 第 860-875 行。

### 7.2 现有 Datalist

| 文件 | 样本数 | 有 Caption | 适用任务 | 来源 |
|------|------:|:---------:|---------|------|
| `eval_transition.json` | 500 | 130 | E2, E4, E5, E6, E10 | 训练集采样，帧数 60-356 |
| `eval_keyframe.json` | 500 | 144 | E3 | 训练集采样，帧数 ≥120 |
| `eval_trajectory.json` | 500 | 130 | E5 | 从 eval_transition 派生 |
| `eval_first_frame_cond.json` | 300 | 300 | E7 | 有 caption 的子集 |
| `eval_loop_animation.json` | 200 | 200 | E8 | 有 caption 的循环动作 |
| `eval_repair.json` | 973 | 0 | E9 | low_quality.json 采样 |
| `eval_repair_focused.json` | 320 | 0 | E9 | 聚焦子集 |
| `eval_transition_with_caption.json` | 300 | 300 | E11 | 有 caption 的 transition |
| `eval_keyframe_with_caption.json` | 300 | 300 | E11 | 有 caption 的 keyframe |
| `eval_datalist_game_20251111.json` | 5692 | 1059 | 补充 | 游戏动画大集 |

### 7.3 Text 来源（E1 / E7 / E8 / E11）

**采用 `data/eval/t2m/251125_yiran_subset.json`** 作为文本提示来源。

该文件包含 **240 条人工审核的英文动作描述**，格式为：

```
"A person walks forward.#60#none#00000001"
 ↑ 文本描述              ↑帧数 ↑条件  ↑ID
```

**特点**：
- 覆盖面广：行走、跑步、运动、舞蹈、日常交互、瑜伽、战斗等
- 帧数范围 30-300（均值 124 帧），与训练分布匹配
- 由一然人工筛选，质量有保证
- 无对应 GT motion — **纯生成任务（E1）只评 FID/Diversity/物理指标，不算 MPJPE**

**各任务使用方式**：

| 任务 | 文本来源 | GT Motion | 说明 |
|------|---------|-----------|------|
| **E1 T2M** | 251125_yiran_subset（240条） | ❌ 无 GT | 纯生成，只评 FID / R-precision / 物理指标 |
| **E7 First-Frame** | eval_first_frame_cond（自带 caption，300条） | ✅ 有 GT | 首帧来自 GT，caption 来自 datalist |
| **E8 Loop** | eval_loop_animation（自带 caption，200条） | ✅ 有 GT | 首尾帧来自 GT，caption 来自 datalist |
| **E11 Caption Comp.** | eval_transition_with_caption / eval_keyframe_with_caption | ✅ 有 GT | caption 来自 datalist |

> **E1 与 E7 的文本来源不同**：E1 用独立的 T2M 提示集（无 GT），E7 用 eval_first_frame_cond 自带的 caption（有 GT）。这是合理的——E1 评纯生成能力，E7 评续写精度。

### 7.4 关键帧数据来源（E3 / E4）

**关键帧不需要额外数据——从 GT motion 的等间距帧自动提取。**

E3 的评测逻辑：

```
输入: GT motion (T, 135/198) + interval K
  ↓
关键帧提取: 取 frame 0, K, 2K, ..., T-1 作为已知帧
  ↓
构建 mask: mask[kf_indices]=0 (keep), 其余=1 (generate)
  ↓
模型推理: 以关键帧为条件，补全中间帧
  ↓
评估: 与 GT 的中间帧对比 MPJPE
```

这与 KIMODO 论文中 keyframe interpolation 的评测方式完全一致：从 test set 的 GT motion 中抽取 keyframe 作为 constraint。

**具体实现**：

| 组件 | 说明 |
|------|------|
| **GT motion NPZ** | 从 `eval_keyframe.json` 的 500 条数据中加载 |
| **关键帧选取** | `build_keyframe_mask(T, D, interval=K)` 自动在 frame 0, K, 2K, ..., T-1 处设 mask=0 |
| **关键帧的"已知值"** | 就是 GT motion 在这些帧的 198-dim 向量（通过 MAN imputation 注入） |
| **评测 GT** | 同一条 GT motion 的非关键帧部分 |

**E3 各 setting 的关键帧密度**：

| Setting | interval | 对于 120 帧序列的关键帧数 | 信息覆盖率 |
|---------|----------|:------------------------:|:---------:|
| A（标准） | 30 | 5 | 4.2% |
| B（稀疏） | 60 | 3 | 2.5% |
| C（密集） | 15 | 9 | 7.5% |
| D（非均匀） | 10-90 随机 | ~4-8 | 变化大 |

**`eval_keyframe.json` 的选择标准**：500 条，全部 ≥120 帧。选择较长的序列是因为关键帧插值在短序列上太简单（30帧序列 + 30帧间隔 = 只有首末两个关键帧）。

**E4 End-Effector 的约束同理**：从 GT motion 做 FK 得到世界坐标关节位置，在指定帧提取指定关节的 3D 位置作为约束值。

### 7.5 数据路径解析流程

```python
# scripts/eval_m2m_all_tasks.py 中已实现的路径解析：
DATA_ROOT = PROJECT_ROOT / "data" / "hymotion_data"
MOTION_ROOTS = [
    DATA_ROOT / "3D" / "20251111" / "motions",  # Game data 在此子目录下
    DATA_ROOT,                                    # Academic/Taobao 等在此
]

def resolve_motion_path(motion_path):
    for root in MOTION_ROOTS:
        full = root / motion_path
        if full.exists():
            return str(full)
    return None
```

> **`tools/eval_m2m_v2_all_tasks.py` 目前使用 `data/motionhub` 作为 data_dir，需要更新为上述多 root 解析逻辑。**

---

## 8. 验证方式

### 8.1 数值验证

每个任务产出指标表格（mean ± std），按以下维度分组汇报：

1. **模型变体**: 4 个 v2 + 可选 v1 基线
2. **Sub-setting**: A/B/C/D
3. **动作类型**: Dance / Locomotion / Combat / Idle / Interaction / Game Special（待标注）

### 8.2 可视化

每个任务选 10 个代表性 case 渲染 video：
- GT（如果有）
- M2M v2 best（选最优模型）
- 基线对比（KIMODO / MoGenDIT / v1）

> 可视化通过 `motion_annot_web` 的 Web 工具查看，或导出 NPZ 后用 Blender 渲染。

### 8.3 统计检验

关键指标做 paired t-test / Wilcoxon signed-rank test：
- uncond_local vs uncond_global → 旋转空间消融
- uncond_local vs caption_local → 文本条件消融
- v2_uncond_local vs v1_uncond_fm_man → v1/v2 代际对比

### 8.4 失败案例分析

每个任务列出 top-5 最差 case（按 MPJPE 或主指标），分析失败原因：
- 序列长度
- 动作类型
- mask 比例
- 旋转空间特异问题

---

## 9. 已知限制与 TODO

### 9.1 当前限制

1. **FID / R-precision 未实现**：E1 的核心指标需要 TMR 特征提取器，需另外集成
2. **Quality checker 集成未完成**：E9 的 QPass rate 需要调用 `quality_check_rules/`，当前脚本未包含
3. **训练尚在进行中**：4 个模型 epoch 86-119（目标 ~500），当前结果为中期快照
4. **Repair mask 未接入 checker**：E9 Setting-A（auto-detect）需要连接 adaptive mask 计算管线
5. **198-dim mask 扩展**：position 通道的 mask 扩展逻辑需验证正确性
6. **数据路径不一致**：`tools/eval_m2m_v2_all_tasks.py` 当前使用 `data/motionhub` 作为 data_dir，应更新为 `data/hymotion_data` + 多 root 解析（参见 §7.5）
7. **E1 T2M 文本加载未接入**：`251125_yiran_subset.json` 格式为 `text#frames#cond#id` 字符串列表，需要专门的解析和 batch 构造逻辑

### 9.2 TODO

- [ ] **修复 data_dir**：`eval_m2m_v2_all_tasks.py` 的 `load_eval_samples()` 改用 `resolve_motion_path()` 多 root 查找
- [ ] **接入 E1 文本数据**：解析 `251125_yiran_subset.json`，为 E1 构造 text-only batch（无 GT motion）
- [ ] 集成 TMR 特征提取器，实现 FID / R-precision / Diversity
- [ ] 连接 quality_check_rules 到 E9 评测流程
- [ ] 标注动作类型分类（dance/locomotion/combat/idle/...），用于分类报告
- [ ] 训练更多 epoch 后重新评测
- [ ] 添加 KIMODO 在 HumanML3D 上的公开数据作为参考值
- [ ] NPZ 输出和 Blender 可视化渲染脚本

---

## 10. 代码文件索引

| 文件 | 用途 |
|------|------|
| `tools/eval_m2m_v2_all_tasks.py` | 主评测脚本（模型加载、推理、报告） |
| `hftrainer/evaluation/motion/m2m_eval_metrics.py` | 15 个指标函数 + 聚合 |
| `hftrainer/evaluation/motion/m2m_eval_tasks.py` | E1-E12 任务定义 + mask builders |
| `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` | M2M ODE 推理管线 |
| `hftrainer/pipelines/motion/differentiable_fk.py` | 可微分 FK（SMPL-22） |
| `hftrainer/evaluation/motion/phys_metrics.py` | 40+ 物理指标（phys_err 等） |
| `hftrainer/evaluation/quality_check_rules/` | 16 个 Quality checker |
| `configs/hymotion_m2m_v2/` | v2 训练配置 |
| `data/eval/hymotion_m2m/` | 评测数据 JSON（动作索引） |
| `data/eval/t2m/251125_yiran_subset.json` | E1 T2M 文本提示（240 条） |
| `data/hymotion_data/` | 动作 NPZ 文件根目录 |
| `scripts/build_eval_datalists.py` | 评测 datalist 构建脚本 |
| `scripts/eval_m2m_all_tasks.py` | v1 综合评测脚本（参考实现） |
