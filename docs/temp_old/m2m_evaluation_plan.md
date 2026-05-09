# HyMotion M2M 全面测评方案

> **版本**: v1.0 | **日期**: 2026-03-30 | **状态**: 草案
> **索引**: 从 `hftrainer/models/motion/CLAUDE.md` §Supported Tasks 引用

---

## 1. 测评目标

为 HyMotion M2M 的 **6 大核心任务** 建立完整的横向（vs 竞品）和纵向（vs 自身消融）对比评估体系：

| # | 任务 | mask 模式 | 对应训练策略 |
|---|------|----------|------------|
| T1 | Motion In-Betweening (MIB) | 首尾帧保留，中间生成 | M3 temporal_contiguous |
| T2 | Motion Prediction | 前缀帧保留，后续生成 | M3 temporal_contiguous |
| T3 | Sparse Keyframe Interpolation | 稀疏关键帧保留，其余生成 | M6 keyframe_sparse |
| T4 | Joint Completion | 部分关节保留，部分关节生成 | M4 joint_contiguous |
| T5 | Motion Repair (Completion) | checker 标记的异常区域生成 | M7 scattered_joint / M1 random_cell |
| T6 | Motion Repair (Editing) | 退化运动作为 reactive 输入 | 编辑范式（reactive ≠ 0） |

---

## 2. 竞品模型

### 2.1 开源模型（可复现）

| 模型 | 机构 | 架构 | 动作表示 | 支持任务 | 代码 | 状态 |
|------|------|------|---------|---------|------|------|
| **KIMODO** | NVIDIA | 2-stage Transformer + DDPM | 333-dim global rot + smooth root (27 joints) | T2M, MIB, End-effector, Trajectory, Multi-prompt | [nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo) | **推理代码已开源**，训练代码未开源 |
| **CondMDI** | — | Diffusion + observation mask | 263-dim HumanML3D repr | MIB, Prediction, Keyframe | [setarehc/CondMDI](https://github.com/setarehc/CondMDI) | 开源 |
| **MotionLab** | — | MotionFlow Transformer | HumanML3D repr | MIB, Prediction, Editing, Stylization | [Diouo/MotionLab](https://github.com/Diouo/MotionLab) | 开源 (ICCV 2025) |
| **OmniControl** | — | MDM + spatial guidance | HumanML3D repr | End-effector, Trajectory, per-joint control | [jiayng01/OmniControl](https://github.com/jiayng01/OmniControl) | 开源 (ICLR 2024) |
| **MoGenDIT** | 内部 | DiT + AdaLN + sliding window | 201-dim local rot + local pos + transl | Repair (denoise, trans_regen) | 内部代码 | 可用 |
| **InterMask** | — | Masked Transformer | HumanML3D repr | MIB, Reaction generation | 开源 | 可对比 reaction |
| **MoMask** | — | Masked Transformer | HumanML3D repr | T2M, MIB | 开源 | 可对比 |

### 2.2 未开源模型（论文数字对比）

| 模型 | 机构 | 核心任务 | 论文关键数字 | 备注 |
|------|------|---------|------------|------|
| **UMO** | Brown/MIT/Meta | MIB, Prediction, Editing, Trajectory, Reaction | MIB MPJPE=8.55mm, T2M FID=9.46 (HumanML3D) | 承诺开源未发布；backbone=HY-Motion-Lite(与我方同) |
| **DART** | — | MIB | LaFAN1 SOTA | 未开源 |

### 2.3 竞品选型建议

按任务选竞品，每个任务至少对比 **2-3 个** 开源方法 + UMO 论文数字：

| 任务 | 必选竞品 | 可选竞品 |
|------|---------|---------|
| T1 MIB | CondMDI, MotionLab, UMO(论文) | MoMask, KIMODO |
| T2 Prediction | CondMDI, MotionLab, UMO(论文) | — |
| T3 Keyframe | CondMDI, KIMODO, UMO(论文) | OmniControl |
| T4 Joint Completion | OmniControl, KIMODO | — |
| T5/T6 Repair | MoGenDIT | — (该任务竞品极少) |

---

## 3. 评估指标

### 3.1 通用质量指标

| 指标 | 定义 | 单位 | 计算方式 | 适用任务 |
|------|------|------|---------|---------|
| **MPJPE** | Mean Per Joint Position Error | mm | FK → 全身 22 关节 L2 距离(pred vs GT) → 平均 | T1-T4 |
| **PA-MPJPE** | Procrustes-Aligned MPJPE | mm | Procrustes 对齐后 L2 | T1-T4 |
| **[P]-MPJPE** | 保留帧/关节的 MPJPE | mm | 仅在 mask=0 区域计算 | T1-T4 |
| **FID** | Fréchet Inception Distance | — | motion feature extractor 嵌入空间距离 | T1, T3 (有 diversity 需求时) |
| **Diversity** | 多次采样结果的多样性 | mm | 同条件 K 次采样两两 MPJPE 平均 | T1, T3 |

### 3.2 运动质量指标

| 指标 | 定义 | 单位 | 计算方式 | 适用任务 |
|------|------|------|---------|---------|
| **Foot Skating** | 接地时脚部水平滑动速度 | cm/s | FK → 脚关节 height<0.05m 时 xz 速度 | 全部 |
| **Jitter** | 加速度抖动 | mm/frame² | `‖p[t+1]-2p[t]+p[t-1]‖` 的平均 | 全部 |
| **Ground Penetration** | 脚部穿地深度 | mm | `min(toe_y, 0)` 的平均绝对值 | 全部 |
| **Boundary Smoothness** | 生成/保留区域边界的平滑度 | mm | 边界帧 ±2 帧内 MPJPE 的变化率 | T1, T2, T4 |
| **Quality Pass Rate** | MotionQualityChecker 通过率 | % | 16 个 checker 全部通过 | 全部 |

### 3.3 任务特定指标

| 指标 | 适用任务 | 定义 |
|------|---------|------|
| **Trajectory Error** | T3 (keyframe), T4 | FK 后关键帧/关键关节位置与约束的 L2 距离 |
| **L2Q** | T1, T2 | L2 Joint Quaternion Error（旋转精度） |
| **NPSS** | T1, T2 | Normalized Power Spectrum Similarity（频域自然度）|
| **Repair Success Rate** | T5, T6 | 修复后 checker 通过率 vs 修复前 |
| **Repair Over-correction** | T5, T6 | 未标记区域被意外修改的程度 |

### 3.4 可选：文本条件指标（如使用 caption）

| 指标 | 定义 |
|------|------|
| **R-Precision** (R@1, R@3) | 生成运动与文本的匹配精度 |
| **MM-Dist** | Motion-text matching distance |

---

## 4. 各任务评估 Setting 详细定义

### 4.1 T1: Motion In-Betweening (MIB)

**任务定义**: 给定首尾若干帧（seed frames），生成中间的过渡运动。

#### Setting 配置

| Setting ID | Seed 帧配置 | 过渡长度 | 总序列长度 | 对标论文 |
|-----------|------------|---------|----------|---------|
| **T1-A** | 首尾各 **1 帧** (frame 0 + frame T-1) | 58 帧 (~2s @30fps) | 60 帧 (2s) | CondMDI short |
| **T1-B** | 首尾各 **1 帧** | 148 帧 (~5s @30fps) | 150 帧 (5s) | CondMDI long |
| **T1-C** | 首尾各 **5 帧** (0.17s) | 50 帧 (~1.7s) | 60 帧 (2s) | UMO setting |
| **T1-D** | 首尾各 **30 帧** (1s) | 120 帧 (4s) | 180 帧 (6s) | 消融实验 Baseline |
| **T1-E** | 首尾各 **30 帧** (1s) | 300 帧 (10s) | 360 帧 (12s) | 长序列极限 |

**说明**:
- T1-A / T1-B 对标 CondMDI 的 standard setting，首尾仅 1 帧 seed，考验模型从极少约束生成自然过渡的能力
- T1-C 对标 UMO 论文 Table 5 的 in-between setting
- T1-D 是我方消融实验的默认 setting（首尾 1s seed 提供充足上下文）
- T1-E 测试长序列生成的稳定性（360 帧 = 12s，接近训练最大长度）

#### mask 构建

```python
# T1-A: 首尾各 1 帧
mask = torch.ones(T, 135)
mask[0, :] = 0          # 第 0 帧保留
mask[T-1, :] = 0        # 最后一帧保留

# T1-D: 首尾各 30 帧
mask = torch.ones(T, 135)
mask[:30, :] = 0         # 前 30 帧保留
mask[-30:, :] = 0        # 后 30 帧保留
```

#### 评估指标
- 主要: **MPJPE**, **[P]-MPJPE**, **Boundary Smoothness**
- 次要: Foot Skating, Jitter, FID, Diversity

---

### 4.2 T2: Motion Prediction

**任务定义**: 给定前缀运动（prefix），预测后续运动。

#### Setting 配置

| Setting ID | Prefix 长度 | 预测长度 | 总序列长度 | 对标论文 |
|-----------|------------|---------|----------|---------|
| **T2-A** | **30 帧** (1s) | 30 帧 (1s) | 60 帧 (2s) | 短期预测 |
| **T2-B** | **30 帧** (1s) | 90 帧 (3s) | 120 帧 (4s) | UMO Table 5 setting |
| **T2-C** | **60 帧** (2s) | 120 帧 (4s) | 180 帧 (6s) | 中长期预测 |
| **T2-D** | **90 帧** (3s) | 270 帧 (9s) | 360 帧 (12s) | 长期极限 |

**说明**:
- T2-A 短期预测（1s→1s），与传统运动预测方法可比
- T2-B 对标 UMO 的 prediction setting
- T2-C/T2-D 考验长程时间一致性

#### mask 构建

```python
# T2-B: prefix 30 帧，预测 90 帧
mask = torch.ones(T, 135)
mask[:30, :] = 0         # 前 30 帧保留
# 后续全部 mask=1
```

#### 评估指标
- 主要: **MPJPE** (预测区域), **[P]-MPJPE**, **Boundary Smoothness**
- 次要: Foot Skating, Jitter
- 注意: Prediction 任务因未来不确定性大，MPJPE 绝对值会随预测长度增长，需要 **分时间段报告**（如 0-1s, 1-2s, 2-3s 的 MPJPE）

---

### 4.3 T3: Sparse Keyframe Interpolation

**任务定义**: 给定稀疏分布的关键帧，补全其余帧。

#### Setting 配置

| Setting ID | 关键帧间隔 | 关键帧数量 | 总序列长度 | 对标论文 |
|-----------|----------|----------|----------|---------|
| **T3-A** | **每 5 帧** 1 个关键帧 | ~24 帧 | 120 帧 (4s) | 密集关键帧 |
| **T3-B** | **每 15 帧** 1 个关键帧 | ~8 帧 | 120 帧 (4s) | CondMDI standard |
| **T3-C** | **每 30 帧** 1 个关键帧 (1s) | ~4 帧 | 120 帧 (4s) | KIMODO keyframe |
| **T3-D** | **每 60 帧** 1 个关键帧 (2s) | ~3 帧 | 180 帧 (6s) | 极稀疏 |
| **T3-E** | **随机** 3-10 个关键帧 | 3-10 帧 | 120 帧 (4s) | 非均匀分布 |

**说明**:
- T3-A 密集关键帧，接近动画师工作流（5 帧 ≈ 0.17s）
- T3-C 对标 KIMODO 的 full-body keyframe 约束
- T3-E 测试随机分布的关键帧，更接近实际使用场景

#### mask 构建

```python
# T3-B: 每 15 帧 1 个关键帧
mask = torch.ones(T, 135)
keyframe_indices = list(range(0, T, 15))  # [0, 15, 30, ...]
for kf in keyframe_indices:
    mask[kf, :] = 0
```

#### 评估指标
- 主要: **MPJPE**, **[P]-MPJPE** (关键帧精度), **Trajectory Error** (关键帧位置偏差)
- 次要: Foot Skating, Jitter, Diversity

---

### 4.4 T4: Joint Completion

**任务定义**: 给定部分关节的运动，补全其余关节。

#### Setting 配置

| Setting ID | 保留关节 | 生成关节 | 说明 | 对标论文 |
|-----------|---------|---------|------|---------|
| **T4-A** | 下半身 (Pelvis, L/R_Hip, L/R_Knee, L/R_Ankle, L/R_Foot, Spine1) | 上半身 (Spine2/3, Neck, Head, L/R_Collar, L/R_Shoulder, L/R_Elbow, L/R_Wrist) | 下→上补全 | 消融实验标准 |
| **T4-B** | 上半身 | 下半身 | 上→下补全 | — |
| **T4-C** | 左半身 | 右半身 | 左→右镜像补全 | — |
| **T4-D** | Root (Pelvis) + Translation | 所有其他关节 | 仅保留根轨迹 | OmniControl trajectory |
| **T4-E** | 全身除双手 (20 joints + transl) | L_Wrist + R_Wrist | End-effector 补全 | KIMODO end-effector |

**说明**:
- T4-A/B 是最常见的上下半身分离测试
- T4-D 测试"从轨迹重建全身运动"，对标 OmniControl 的 trajectory following
- T4-E 测试 end-effector（手腕）补全，是动画制作中的常见需求

#### mask 构建

```python
# T4-A: 下半身保留，上半身生成
# 关节分组（基于 135-dim layout）:
LOWER_BODY_JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 10, 11]  # joint indices
UPPER_BODY_JOINTS = [6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

mask = torch.zeros(T, 23)  # (T, 23) joint group grid
for j in UPPER_BODY_JOINTS:
    mask[:, j+1] = 1       # +1 因为 col 0 = translation
mask[:, 0] = 0             # translation 保留
expanded_mask = expand_grid_to_mask(mask)  # (T, 23) -> (T, 135)
```

#### 评估指标
- 主要: **MPJPE** (生成关节), **[P]-MPJPE** (保留关节)
- 次要: Jitter (生成区域), 关节间连贯性（spine chain 连续性）

---

### 4.5 T5: Motion Repair (Completion 模式)

**任务定义**: 通过 quality checker 检测运动缺陷位置，将缺陷区域 mask=1 后由模型重新生成。

#### Setting 配置

| Setting ID | 缺陷类型 | mask 来源 | 说明 |
|-----------|---------|----------|------|
| **T5-A** | Checker 检测真实缺陷 | `MotionQualityChecker` 16 checker → adaptive mask | 全自动修复流程 |
| **T5-B** | 合成 foot sliding | FK → height < 0.05m & xz_vel > 0.5cm/s 的帧/关节 mask | 定向修复 foot sliding |
| **T5-C** | 合成 joint jump | 相邻帧位移 > 阈值的帧/关节 mask | 定向修复跳变 |
| **T5-D** | 合成 arm penetration | 手臂穿透躯干的帧/关节 mask | 定向修复穿透 |
| **T5-E** | 随机 scattered mask | M7 策略生成的随机 scattered mask（模拟 checker mask 分布） | 泛化能力测试 |

#### 评估数据

使用 `data/hymotion_m2m_refine_data/data_quality_list/low_quality.json` 中的低质量样本：
- 每个样本有 checker 标记的具体缺陷和位置
- 修复前/后对比 checker 通过率

#### 评估指标
- 主要: **Repair Success Rate** (修复后 checker 通过率), **Repair Over-correction** (未标记区域变化 MPJPE)
- 次要: 修复前后 Foot Skating 变化, Jitter 变化, 整体 Quality Pass Rate 变化
- 特殊: **Repair Precision** = 缺陷区域被修复的比例; **Repair Recall** = 修复后无新缺陷引入的比例

---

### 4.6 T6: Motion Repair (Editing 模式)

**任务定义**: 将退化运动作为 reactive 通道输入，模型在退化基础上进行修正。

#### Setting 配置

| Setting ID | 退化类型 | reactive 输入 | 说明 |
|-----------|---------|-------------|------|
| **T6-A** | 高斯噪声 | 原始运动 + N(0, σ²), σ∈{0.01, 0.05, 0.1} | 去噪能力 |
| **T6-B** | 关节冻帧 | 随机 2-5 关节冻结 5-20 帧 | 冻帧修复 |
| **T6-C** | 位移漂移 | translation 随机 offset 0.1-0.5m | 位移纠正 |
| **T6-D** | MoGenDIT 8 种合成缺陷 | MoGenDIT corruptor 生成 | 与 MoGenDIT 对标 |

#### 与 T5 的区别
- T5: mask=1 区域 reactive=0（从零生成），模型完全不看退化运动
- T6: mask=1 区域 reactive=退化运动值，模型在退化基础上修正

#### 评估指标
- 主要: **MPJPE** (修复后 vs GT), **Repair Success Rate**
- 对比: T5 vs T6 在相同缺陷上的修复效果差异

---

## 5. 测试数据

### 5.1 数据集选择

| 数据集 | 用途 | 大小 | 说明 |
|--------|------|------|------|
| **HumanML3D test split** | T1-T4 横向对比（与竞品统一） | 4,646 序列 | 业界标准 benchmark，所有竞品论文均在此评测 |
| **MotionHub val split** | T1-T6 纵向评估（内部）| ~200-500 序列 | 从 hymotion_400h val split 采样，更多样 |
| **Low-quality set** | T5-T6 修复评估 | ~数千序列 | `data/hymotion_m2m_refine_data/data_quality_list/low_quality.json` |
| **LaFAN1** | T1 横向对比（传统 MIB benchmark） | 标准 test split | 传统 MIB 方法常用，30/45 帧过渡 |

### 5.2 HumanML3D 测试集准备

**为什么必须用 HumanML3D**: 几乎所有竞品（CondMDI, MotionLab, UMO, OmniControl, MoMask）都在 HumanML3D test split 上报告数字，是横向对比的唯一可行基准。

**表示转换**: HumanML3D 原生使用 263-dim 表示（含 root velocity, local joint positions 等），需要转换到我方 135-dim 表示：
1. 从 HumanML3D 提取 SMPL axis-angle 参数
2. 通过 `process_smplx_pose` 转换为 135-dim (abs transl + rot6d row-major)
3. 使用训练时相同的 Mean/Std normalize

**注意**: HumanML3D 使用 20fps（部分版本），我方使用 30fps。需要统一到 30fps 或在评估时做 fps 转换。

### 5.3 测试集采样策略

| 测试集 | 采样数量 | 采样策略 |
|--------|---------|---------|
| HumanML3D (横向对比) | **全量** test split (~4646) | 完整评估，与竞品数字可比 |
| MotionHub (纵向/消融) | **200 序列** | 从 val split 随机采样，覆盖多种运动类型 |
| Low-quality (修复) | **100 序列** | 从 low_quality.json 采样，按缺陷类型分层 |

### 5.4 测试集构建脚本（TODO）

```bash
# 1. 准备 HumanML3D 测试集（转换到 135-dim）
python scripts/prepare_humanml3d_test.py \
    --humanml3d_root <path_to_humanml3d> \
    --output data/eval/humanml3d_test_135dim/

# 2. 准备 MotionHub 评估子集
python scripts/prepare_motionhub_eval.py \
    --val_split data/motionhub/val.json \
    --num_samples 200 \
    --output data/eval/motionhub_eval_200/

# 3. 准备低质量修复测试集
python scripts/prepare_repair_eval.py \
    --quality_list data/hymotion_m2m_refine_data/data_quality_list/low_quality.json \
    --num_samples 100 \
    --output data/eval/repair_eval_100/
```

---

## 6. 竞品评估执行方案

### 6.1 KIMODO（开源推理）

**环境**: `ref_repo/KIMODO/kimodo/` 已有代码

**评估任务**:
- T1 MIB: 首尾帧 FullBodyConstraintSet → imputation
- T3 Keyframe: 稀疏帧 FullBodyConstraintSet → imputation
- T4-E End-effector: EndEffectorConstraintSet (hands/feet)

**表示转换**: 需要将 HumanML3D/MotionHub 数据转到 KIMODO 的 333-dim global rotation 表示。KIMODO 提供了 retarget 工具。

**推理配置**: DDIM 100 steps, separated CFG (w_text=2.0, w_constr=2.0)

### 6.2 CondMDI（开源）

**评估任务**:
- T1 MIB: 首尾帧 observation mask
- T2 Prediction: prefix observation mask
- T3 Keyframe: 稀疏帧 observation mask

**表示**: 263-dim HumanML3D repr，无需转换（直接在 HumanML3D 上评测）

**结果转换**: CondMDI 输出 263-dim → 通过 FK 得到关节位置 → 计算 MPJPE

### 6.3 MotionLab（开源）

**评估任务**:
- T1 MIB: Motion-Condition-Motion 范式
- T2 Prediction: prefix conditioning

**结果转换**: 同 CondMDI

### 6.4 OmniControl（开源）

**评估任务**:
- T4-D Trajectory: pelvis trajectory control
- T4-E End-effector: per-joint spatial guidance

### 6.5 MoGenDIT（内部）

**评估任务**:
- T5 Repair (Completion): denoise / ada_denoise 模式
- T6 Repair (Editing): 退化输入修复

**Pipeline**: `hftrainer/pipelines/motion/mogendit_pipeline.py`

### 6.6 UMO（论文数字）

**评估任务**: 直接引用论文 Table 5-8 的数字
- T1 MIB: MPJPE=8.55mm (HumanML3D)
- T2 Prediction: 论文 Table 5
- Editing: 论文 Table 7 (MotionFix)
- Trajectory: 论文 Table 8

**注意**: UMO 使用 201-dim 表示（含 local joint positions），与我方 135-dim 不同。MPJPE 计算均基于 FK 后的 3D 关节位置，因此可直接对比。

---

## 7. 统一评估流程

### 7.1 评估 Pipeline

```
输入: GT motion (NPZ/pkl) + mask definition
  │
  ├── 1. 表示转换（如需）
  │     HumanML3D 263-dim → 135-dim (我方)
  │     HumanML3D 263-dim → 333-dim (KIMODO)
  │     ...
  │
  ├── 2. Mask 构建
  │     按 Setting (T1-A ~ T6-D) 生成 mask
  │
  ├── 3. 推理
  │     各模型按各自 pipeline 推理
  │     统一输出: predicted motion (NPZ)
  │
  ├── 4. FK 转换
  │     所有模型输出 → SMPL FK → 22 关节 3D 位置 (T, 22, 3)
  │     使用统一 SMPL body model
  │
  └── 5. 指标计算
        统一计算所有指标（MPJPE, Foot Skating, Jitter, ...）
```

### 7.2 关键: 统一 FK 和指标计算

**所有模型的输出都必须经过相同的 FK pipeline** 转换到 3D 关节位置后再计算指标，避免因表示差异导致的不公平对比。

```python
# 统一指标计算接口
class MotionEvaluator:
    def __init__(self, body_model_path):
        self.body_model = load_smpl(body_model_path)

    def evaluate(self, pred_joints_3d, gt_joints_3d, mask, task_type):
        """
        Args:
            pred_joints_3d: (T, 22, 3) — FK 后的预测关节位置
            gt_joints_3d: (T, 22, 3) — FK 后的 GT 关节位置
            mask: (T, 22) — joint-frame mask (0=preserved, 1=generated)
            task_type: str — 'mib', 'prediction', 'keyframe', 'joint', 'repair'
        Returns:
            dict of metrics
        """
        metrics = {}
        gen_mask = mask.bool()
        pres_mask = ~mask.bool()

        # MPJPE (generated region)
        if gen_mask.any():
            metrics['MPJPE'] = (pred_joints_3d[gen_mask] - gt_joints_3d[gen_mask]).norm(dim=-1).mean() * 1000  # mm

        # [P]-MPJPE (preserved region)
        if pres_mask.any():
            metrics['P_MPJPE'] = (pred_joints_3d[pres_mask] - gt_joints_3d[pres_mask]).norm(dim=-1).mean() * 1000

        # Foot Skating, Jitter, Ground Penetration, Boundary Smoothness...
        metrics.update(self._compute_quality_metrics(pred_joints_3d))

        return metrics
```

---

## 8. 结果呈现

### 8.1 主表: 横向竞品对比（HumanML3D test）

#### Table 1: Motion In-Betweening (T1)

| Method | Setting | MPJPE↓ | [P]-MPJPE↓ | Foot Skating↓ | Jitter↓ | Boundary Smooth↓ |
|--------|---------|--------|-----------|--------------|---------|-----------------|
| CondMDI | T1-A | — | — | — | — | — |
| MotionLab | T1-A | — | — | — | — | — |
| UMO | T1-C | 8.55 | 0.95 | — | — | — |
| KIMODO | T1-D | — | — | — | — | — |
| **M2M (Ours)** | T1-A | — | — | — | — | — |
| **M2M (Ours)** | T1-C | — | — | — | — | — |
| **M2M (Ours)** | T1-D | — | — | — | — | — |

#### Table 2: Motion Prediction (T2)

| Method | Setting | MPJPE↓ (0-1s) | MPJPE↓ (1-2s) | MPJPE↓ (2-3s) | [P]-MPJPE↓ | Foot Skating↓ |
|--------|---------|-------------|-------------|-------------|-----------|--------------|
| CondMDI | T2-B | — | — | — | — | — |
| MotionLab | T2-B | — | — | — | — | — |
| UMO | T2-B | — | — | — | — | — |
| **M2M (Ours)** | T2-B | — | — | — | — | — |

#### Table 3: Sparse Keyframe Interpolation (T3)

| Method | Interval | MPJPE↓ | [P]-MPJPE↓ | Traj Error↓ | Diversity↑ |
|--------|----------|--------|-----------|------------|-----------|
| CondMDI | 15 frames | — | — | — | — |
| KIMODO | 30 frames | — | — | — | — |
| **M2M (Ours)** | 15 frames | — | — | — | — |
| **M2M (Ours)** | 30 frames | — | — | — | — |

#### Table 4: Joint Completion (T4)

| Method | Setting | MPJPE↓ (gen joints) | [P]-MPJPE↓ (pres joints) | Jitter↓ |
|--------|---------|-------------------|------------------------|---------|
| OmniControl | T4-A | — | — | — |
| KIMODO | T4-E | — | — | — |
| **M2M (Ours)** | T4-A | — | — | — |
| **M2M (Ours)** | T4-E | — | — | — |

#### Table 5: Motion Repair (T5 + T6)

| Method | Mode | Repair Success↑ | Over-correction↓ | MPJPE↓ | ΔFoot Skating |
|--------|------|----------------|------------------|--------|---------------|
| MoGenDIT | denoise | — | — | — | — |
| MoGenDIT | ada_denoise | — | — | — | — |
| **M2M (T5)** | completion | — | — | — | — |
| **M2M (T6)** | editing | — | — | — | — |

### 8.2 消融表

继承 `ref_repo/m2m_ablation_experiments.md` 的评估框架，但使用本文档定义的标准化 settings（T1-D, T2-B, T4-A, T5-A）。

---

## 9. 实施优先级

### Phase 1: 基础设施（1 周）

| 优先级 | 任务 | 输出 |
|--------|------|------|
| P0 | 实现统一 `MotionEvaluator` 类 | `hftrainer/evaluation/motion_evaluator.py` |
| P0 | 准备 HumanML3D 测试集（表示转换脚本） | `scripts/prepare_humanml3d_test.py` |
| P0 | 实现 T1-T4 的 mask 构建工具 | `hftrainer/evaluation/mask_builder.py` |
| P1 | 实现 T5/T6 的 repair 评估流程 | `scripts/eval_m2m_repair.py` 扩展 |

### Phase 2: 竞品复现（1-2 周）

| 优先级 | 任务 | 输出 |
|--------|------|------|
| P0 | KIMODO 推理 pipeline 跑通 + HumanML3D 评测 | KIMODO 数字 |
| P0 | CondMDI 推理 pipeline 跑通 + HumanML3D 评测 | CondMDI 数字 |
| P1 | MotionLab 推理 pipeline 跑通 | MotionLab 数字 |
| P1 | OmniControl 推理 pipeline 跑通 (T4 任务) | OmniControl 数字 |
| P2 | MoGenDIT 修复评测 (T5/T6) | MoGenDIT 数字 |

### Phase 3: 我方模型评测（1 周）

| 优先级 | 任务 | 输出 |
|--------|------|------|
| P0 | M2M 在 T1-T4 所有 settings 上评测 | M2M 数字 |
| P0 | M2M 在 T5-T6 修复评测 | M2M 修复数字 |
| P1 | 消融实验全量评测 | 消融结果表 |

### Phase 4: 报告汇总（3 天）

| 优先级 | 任务 | 输出 |
|--------|------|------|
| P0 | 填入所有结果表格 | 本文档更新 |
| P0 | 撰写分析结论 | 优劣势总结 |
| P1 | 可视化对比（渲染视频） | 代表性 case 对比视频 |

---

## 10. 开放问题

1. **HumanML3D 表示转换精度**: 263-dim → 135-dim 存在信息损失（丢失 velocity, joint position），GT 重建精度需要验证
2. **fps 不一致**: HumanML3D 部分数据为 20fps，我方为 30fps，需要统一处理
3. **FID 计算**: 需要训练/使用 motion feature extractor（如 T2M-GPT 的 encoder），目前未实现
4. **文本条件 vs 无条件**: 竞品多为 text-conditioned，我方 M2M 当前主要为 unconditioned completion。需要决定是否在 text-conditioned 模式下对比
5. **长序列评估**: 360 帧评估需要确认推理 ODE 在长序列上的稳定性
6. **Repair 评估基准缺失**: Motion repair 任务缺乏公认 benchmark，T5/T6 的评估主要是内部纵向对比

---

## 参考文献

- KIMODO: Rempe et al., "Scaling Controllable Human Motion Generation", 2026
- UMO: Cong et al., "Unified In-Context Learning Unlocks Motion Foundation Model Priors", arXiv:2603.15975, 2026
- CondMDI: Cohan et al., "Flexible Motion In-betweening with Diffusion Models", arXiv:2405.11126, 2024
- MotionLab: "MotionLab: Unified Human Motion Generation and Editing", ICCV 2025
- OmniControl: "OmniControl: Control Any Joint at Any Time for Human Motion Generation", ICLR 2024
- MoGenDIT: 内部扩散修复框架
- LaFAN1: Harvey et al., "Robust Motion In-betweening", SIGGRAPH 2020
- HumanML3D: Guo et al., "Generating Diverse and Natural 3D Human Motions from Text", CVPR 2022
