# HyMotion M2M 综合测评方案

> **版本**: v5.0 | **日期**: 2026-04-01 | **状态**: 正式
> **上一版**: `docs/temp/m2m_evaluation_plan.md`（v1.0, 草案）
> **索引**: 从 `hftrainer/models/motion/CLAUDE.md` §Supported Tasks 引用
> **Datalist 路径**: `data/eval/hymotion_m2m/`

---

## 1. 测评目标

为 HyMotion M2M 建立完整的横向（vs 竞品）评估体系，覆盖 **9 个核心评测任务**：

| #   | 任务                                           | mask 模式              | 是否需要文本 | 对应训练策略               |
| --- | -------------------------------------------- | -------------------- | ------ | -------------------- |
| **T0**  | **Text-to-Motion (纯文本生成)**                    | **全 mask (all=1)**      | **必须** | **M5 full_mask**       |
| T1  | Motion Transition (两段过渡)                      | 首尾帧保留，中间生成           | 可选     | M3 temporal_contiguous |
| T2  | Sparse Keyframe Interpolation (不同密度)          | 稀疏关键帧保留              | 可选     | M6 keyframe_sparse   |
| T3  | First-Frame Conditioned Generation (首帧+文本)    | 首帧保留，其余生成            | **必须** | M3 temporal_contiguous |
| T4  | Loop Animation (循环动画+文本)                      | 首尾帧保留(相同)，中间生成       | **必须** | M3 temporal_contiguous |
| T5  | Motion Prediction                             | 前缀保留，后续生成            | 可选     | M3 temporal_contiguous |
| T6  | Joint Completion                              | 部分关节保留               | 可选     | M4 joint_contiguous  |
| T7  | Motion Repair                                 | 缺陷区域生成               | 可选     | M7 scattered_joint   |
| **T8**  | **Trajectory-Based Generation (轨迹条件生成)**      | **Root transl 保留，其余生成** | **必须** | **M4 joint_contiguous** |

> **T0 说明**: T0 是专门为验证 **T2M + M2M 混合训练策略是否导致 T2M 能力退化** 而设的对比任务。M2M 的 M5 full_mask（占训练比重 5%）退化为纯 T2M，需要与 HunyuanMotion T2M 1.0（Cascade model）和 KIMODO 的纯 T2M 能力做横向对比，量化混合训练的代价。
>
> **T8 说明**: T8 是给定文本描述 + 根节点运动轨迹（translation），生成符合轨迹约束的完整动作。这是 KIMODO 的核心对比任务，验证 M2M 的 joint-level mask 是否可以自然支持 trajectory conditioning。

---

## 2. 竞品模型

> **竞品选择原则**: 只对比具备 MIB (Motion In-Betweening) / Motion Completion 能力的**商业级产品和闭源竞品**。开源学术模型（MDM, MLD, MoMask, T2M-GPT 等）的能力已被我方方案全面超越，不再作为横向对比对象，仅在 T0 中保留论文数字作为参考水位。

### 2.1 商业软件/闭源产品（具备 MIB 能力）

| 产品/模型 | 机构 | MIB 核心能力 | 获取方式 | 评测可行性 | 备注 |
|----------|------|------------|---------|-----------|------|
| **Autodesk Maya 2026+** | Autodesk | ML-based Motion In-Betweening, Keyframe Interpolation, Pose-to-Pose Completion | Maya 2026 内置 ML Motion 功能 | 导出 FBX → retarget SMPL → 定量对比 | Maya 2026 新增 AI Motion 功能，支持稀疏关键帧补间、过渡生成 |
| **VIVISE** | 腾讯/内部 | Motion In-Betweening, Motion Editing, Trajectory-guided Generation | 内部获取 | 需协调内部接口 | 腾讯内部具备 MIB 能力的商业化项目，需确认是否可提供评测接口 |
| **KIMODO** | NVIDIA | T2M + Keyframe + End-effector + Trajectory + Multi-prompt | [GitHub: nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo) + `pip install kimodo` | 本地 Python API 定量评测 | 已开源，推理代码完整可用；**T8 trajectory 的主要对比对象** |
| **HunyuanMotion 1.0** (Cascade T2M) | 腾讯/内部 | T2M（不支持 MIB） | [Playground](https://hy-motion.ai/playground) + 内部推理 | 本地定量 + Web 定性 | 仅用于 T0 T2M 能力退化验证，不参与 T1-T8 对比 |
| **MoGenDIT** | 内部(chengxuzuo) | Motion Repair (denoise, trans_regen) | 内部 pipeline | 直接调用 | 仅用于 T7 Repair 对比 |

### 2.2 未开源但有论文数据可参考的模型

| 模型 | 机构 | 核心能力 | 论文关键数字 | 状态 |
|------|------|---------|-----------|------|
| **UMO** | Brown/MIT/Meta/MPI/HKU | MIB, Prediction, Editing, Trajectory | MIB MPJPE=8.55mm, T2M FID=9.46 | 承诺开源未发布；backbone=HY-Motion-Lite(与我方同) |

### 2.3 学术模型论文数字（仅作参考水位，不做横向对比）

> 以下模型**不需要本地部署复现**，仅引用论文报告数字作为 T0 的绝对水平参考。

| 模型 | 发表 | T2M FID↓ | T2M R@1↑ | 备注 |
|------|------|----------|----------|------|
| MoMask | CVPR 2024 | 0.045 | 0.521 | HumanML3D SOTA (学术) |
| T2M-GPT | CVPR 2023 | 0.116 | 0.491 | — |
| MLD | CVPR 2023 | 0.473 | 0.390 | — |
| MDM | ICLR 2023 | 0.544 | 0.320 | — |
| PackDiT | 2025 | 0.106 | — | — |
| UMO | 2026 | 9.46 | — | HumanML3D（不同评测协议） |

### 2.4 各任务竞品选型

| 任务 | 必选竞品 | 可选竞品 | 论文数字参考 |
|------|---------|---------|-----------|
| **T0 T2M** | **HunyuanMotion 1.0, KIMODO** | — | MoMask, T2M-GPT, MLD, MDM (学术水位) |
| T1 Transition | **KIMODO, Maya 2026** | VIVISE | UMO(论文) |
| T2 Keyframe | **KIMODO, Maya 2026** | VIVISE | UMO(论文) |
| T3 First-Frame+Text | **KIMODO** | Maya 2026 | UMO(论文) |
| T4 Loop Animation | — (我方独有) | — | — |
| T5 Prediction | **KIMODO** | Maya 2026 | UMO(论文) |
| T6 Joint Completion | **KIMODO** | — | — |
| T7 Repair | **MoGenDIT** | — | — |
| **T8 Trajectory** | **KIMODO** | VIVISE | UMO(论文) |

**竞品说明**:
- **Maya 2026**: Autodesk 最新版本内置 ML Motion In-Betweening 功能，是动画师实际使用的商业工具，具有极强的行业参考价值。评测时导出 FBX → retarget 到 SMPL 22-joint → 统一指标计算。
- **VIVISE**: 腾讯内部项目，具备 MIB 能力。暂无公开资料，需内部协调评测接口。如无法获取推理接口，标记为 "待评测"。
- **KIMODO**: NVIDIA 开源，T2M + MIB + Trajectory 全能力覆盖，**T8 的核心对比对象**。本地 `pip install kimodo` 定量评测。
- **HunyuanMotion 1.0**: 仅用于 T0 T2M 退化验证，不支持 completion 任务。
- **学术开源模型 (MDM, MoMask, CondMDI 等)**: 已确认我方方案全面超越，不再作为横向对比对象。T0 中保留论文 FID/R@k 数字仅作为绝对水平参考。

---

## 3. 评估指标

### 3.1 通用质量指标

| 指标 | 定义 | 单位 | 适用任务 |
|------|------|------|---------|
| **MPJPE** | Mean Per Joint Position Error (FK→22关节L2) | mm | T1-T6, T8 |
| **PA-MPJPE** | Procrustes-Aligned MPJPE | mm | T1-T6, T8 |
| **[P]-MPJPE** | 保留帧/关节的 MPJPE | mm | T1-T6, T8 |
| **FID** | Fréchet Inception Distance (motion feature space) | — | T0-T4, T8 |
| **Diversity** | 多次采样两两 MPJPE 均值 | mm | T0-T4, T8 |

### 3.2 运动物理质量指标

| 指标 | 定义 | 单位 | 适用任务 |
|------|------|------|---------|
| **Foot Skating** | 接地时脚部水平滑动 | cm/s | 全部 |
| **Jitter** | 加速度抖动 (jerk) | mm/frame² | 全部 |
| **Ground Penetration** | 脚部穿地深度 | mm | 全部 |
| **Boundary Smoothness** | 生成/保留区域边界帧过渡平滑度 | mm | T1, T2, T4, T5, T6 |
| **Quality Pass Rate** | MotionQualityChecker 16-checker 通过率 | % | 全部 |

### 3.3 任务特定指标

| 指标 | 适用任务 | 定义 |
|------|---------|------|
| **Trajectory Error** | T2, T4, T8 | FK 后关键帧位置与约束的 L2 |
| **Trajectory ADE** | T8 | Average Displacement Error (root transl vs GT) |
| **Trajectory FDE** | T8 | Final Displacement Error (末帧 root 位置误差) |
| **L2Q** | T1, T5 | L2 Joint Quaternion Error |
| **NPSS** | T1, T5 | Normalized Power Spectrum Similarity |
| **Repair Success Rate** | T7 | 修复后 checker 通过率 |
| **Repair Over-correction** | T7 | 未标记区域变化 MPJPE |
| **Loop Continuity Error** | T4 | 首尾帧的 MPJPE (应为 ~0) |
| **Text-Motion R-Precision** | T0, T3, T4, T8 | R@1, R@3 |
| **MM-Dist** | T0, T3, T4, T8 | Motion-text matching distance |

---

## 4. 各任务评估 Setting 详细定义

### 4.0 T0: Text-to-Motion (纯文本生成 — T2M 能力退化验证)

**任务定义**: 给定文本描述，从零生成完整动作序列。M2M 通过 `mask = all 1`（M5 full_mask）退化为纯 T2M。

**核心目的**: 量化 T2M + M2M 混合训练策略对 T2M 能力的影响。M5 full_mask 在训练中仅占 5%，需要验证这是否导致 T2M 生成质量显著下降。

**对比对象**:
- **HunyuanMotion T2M 1.0** (Cascade model): 同团队的纯 T2M baseline，**主要对比对象**
- **KIMODO**: NVIDIA 开源，Phase 1 纯 T2M 训练，T2M 质量有保障
- **MDM / MLD / T2M-GPT / MoMask / MotionGPT / PackDiT**: HumanML3D 上的经典/SOTA baseline（论文数字参考 + 可选复现）

#### Setting 配置

| Setting ID | 生成长度 | 文本类型 | 说明 |
|-----------|---------|---------|------|
| **T0-A** | 60 帧 (2s) | simple_caption | 短序列 |
| **T0-B** | 90 帧 (3s) | simple_caption | 中序列 |
| **T0-C** | 120 帧 (4s) | simple_caption | 长序列 |
| **T0-D** | 随机 60-150 帧 | simple_caption | 混合长度（与 HumanML3D 评测对齐）|
| **T0-E** | 240-600 帧 (8-20s) | 组合文本 | 长动作生成（从 yiran_subset 中组合 2-4 条文本）|

**mask 构建**:
```python
# T0: 全 mask，退化为纯 T2M
mask = torch.ones(T, 135)  # 全部生成
# src_motion 全零（无条件）
src_motion = torch.zeros(T, 135)
```

**评估指标**:
- **主要**: FID↓, R@1↑, R@3↑, MM-Dist↓ (文本-动作匹配)
- **多样性**: Diversity, MultiModality
- **质量**: Foot Skating, Jitter, Ground Penetration, Quality Pass Rate
- 如果使用 HumanML3D 评测集，可直接与论文数字对比

**评测数据**:
- **内部游戏数据**: `eval_t2m.json` (300 条, 从 eval_datalist_game_20251111.json 中按文本质量筛选)
- **yiran 测评文本**: `data/eval/t2m/251125_yiran_subset.json` (240 条人工编写测评文本，格式: `"text#frames#none#id"`，覆盖多种动作类型)
- **长动作组合**: 从 `251125_yiran_subset.json` 中选取多条文本进行拼接组合，测试长序列 T2M 生成（如 "A person walks forward" + "A person turns left" + "A person sits down"）
- **公开数据 (可选)**: HumanML3D test set（需要做表示转换 263-dim → 135-dim），以便与论文报告数字直接对比

**关键分析维度**:
1. M2M (mask=all 1) vs HunyuanMotion T2M 1.0：**混合训练代价**（核心问题）
2. M2M vs KIMODO T2M：跨方案对比
3. M2M vs HumanML3D baseline 论文数字：绝对水平定位

**使用 datalist**:
- `eval_t2m.json` (300 条游戏数据, 全部有文本, 从 eval_datalist_game_20251111.json 中按文本质量筛选)
- `data/eval/t2m/251125_yiran_subset.json` (240 条 yiran 人工编写文本, 格式 `"text#frames#none#id"`)
- T0-E 长动作组合：从 yiran_subset 中随机抽取 2-4 条文本拼接，按各文本指定帧数累加为总长度

---

### 4.1 T1: Motion Transition (两段动作过渡)

**任务定义**: 给定两段动作（A 的尾部 + B 的头部），生成中间的自然过渡。对应实际应用中的动作拼接。

#### Setting 配置

| Setting ID | Seed 帧配置 | 过渡长度 | 总序列长度 | 说明 |
|-----------|-----------|---------|---------|------|
| **T1-A** | 首尾各 **1 帧** | 58 帧 (~2s) | 60 帧 | CondMDI short setting |
| **T1-B** | 首尾各 **1 帧** | 148 帧 (~5s) | 150 帧 | CondMDI long setting |
| **T1-C** | 首尾各 **5 帧** (0.17s) | 50 帧 | 60 帧 | UMO setting |
| **T1-D** | 首尾各 **10 帧** (0.33s) | 40 帧 | 60 帧 | 短过渡 |
| **T1-E** | 首尾各 **30 帧** (1s) | 120 帧 (4s) | 180 帧 | 消融 Baseline |
| **T1-F** | 首尾各 **30 帧** (1s) | 300 帧 (10s) | 360 帧 | 长序列极限 |

**数据构造方法**:
- 从测试集中选取 >= 60 帧的动作序列
- 对于 T1-A/B/C/D：取完整序列，首尾帧做 seed
- 对于 T1-E/F：如序列不够长，拼接两段不同序列的尾部/头部，模拟真实过渡场景

**mask 构建**:
```python
# T1-C: 首尾各5帧
mask = torch.ones(T, 135)
mask[:5, :] = 0          # 前5帧保留
mask[-5:, :] = 0         # 后5帧保留
```

**评估指标**: MPJPE, [P]-MPJPE, Boundary Smoothness, Foot Skating, Jitter

**文本**: 可选。有文本时使用 simple_caption；无文本时使用 null embedding。

**使用 datalist**: `eval_transition.json` (500 条, >= 60 帧) + `eval_transition_with_caption.json` (300 条, 有文本)

---

### 4.2 T2: Sparse Keyframe Interpolation (不同密度的关键帧补间)

**任务定义**: 给定 N 个 keyframe，补间成完整动作。重点测评不同 keyframe 数量/频率的效果，尤其是 keyframe 非常稀疏的场景。

#### Setting 配置

| Setting ID | 关键帧间隔 | 每120帧的关键帧数 | 总序列长度 | 稀疏程度 | 说明 |
|-----------|----------|------------|---------|------|------|
| **T2-A** | **每 5 帧** | ~24 帧 | 120 帧 | 密集 | 动画师工作流 |
| **T2-B** | **每 10 帧** | ~12 帧 | 120 帧 | 中密 | 常规补间 |
| **T2-C** | **每 15 帧** | ~8 帧 | 120 帧 | 中稀 | CondMDI standard |
| **T2-D** | **每 30 帧** (1s) | ~4 帧 | 120 帧 | 稀疏 | KIMODO keyframe |
| **T2-E** | **每 60 帧** (2s) | ~3 帧 | 180 帧 | 极稀疏 | 极限测试 |
| **T2-F** | **每 90 帧** (3s) | ~2 帧 | 180 帧 | 超极稀疏 | 退化为近 MIB |
| **T2-G** | **随机 3-10 帧** | 3-10 帧 | 120 帧 | 非均匀 | 实际使用场景 |
| **T2-H** | **仅首尾 + 中间1帧** | 3 帧 | 120 帧 | 最稀疏 | 3-keyframe极限 |

**关键帧选择策略**:
- T2-A ~ T2-F: 均匀间隔 + 首尾帧强制包含
- T2-G: 随机采样（固定 seed 保证可复现）
- T2-H: 仅 frame[0], frame[T//2], frame[T-1]

**mask 构建**:
```python
# T2-D: 每30帧1个关键帧
mask = torch.ones(T, 135)
keyframe_indices = list(range(0, T, 30))
if (T-1) not in keyframe_indices:
    keyframe_indices.append(T-1)
for kf in keyframe_indices:
    mask[kf, :] = 0
```

**评估指标**: MPJPE, [P]-MPJPE, Trajectory Error, Foot Skating, Jitter
- **重点**: 按间隔分段报告 MPJPE，观察随稀疏度增加的衰减曲线

**使用 datalist**: `eval_keyframe.json` (500 条, >= 120 帧) + `eval_keyframe_with_caption.json` (300 条)

---

### 4.3 T3: First-Frame Conditioned Generation (首帧条件生成，必须有文本)

**任务定义**: 给定真实数据的第一帧作为初始姿态，结合文本描述生成完整动作序列。

**必须有文本**: 仅首帧条件 + 全 mask 退化为无引导生成，文本是决定运动语义的唯一信号。

#### Setting 配置

| Setting ID | 首帧保留 | 生成长度 | 说明 |
|-----------|--------|---------|------|
| **T3-A** | 第 0 帧 | 59 帧 (2s) | 短序列 |
| **T3-B** | 第 0 帧 | 89 帧 (3s) | 中序列 |
| **T3-C** | 第 0 帧 | 119 帧 (4s) | 长序列 |
| **T3-D** | 前 5 帧 | 115 帧 (3.8s) | 多帧 seed |

**mask 构建**:
```python
# T3-A: 仅保留第0帧
mask = torch.ones(60, 135)
mask[0, :] = 0  # 首帧保留
```

**评估指标**:
- **主要**: Text-Motion R-Precision (R@1, R@3), MM-Dist
- **质量**: Foot Skating, Jitter, Quality Pass Rate
- **首帧精度**: [P]-MPJPE (第0帧 vs GT)
- **FID**, Diversity

**使用 datalist**: `eval_first_frame_cond.json` (300 条, 全部有文本)

---

### 4.4 T4: Loop Animation (循环动画生成，必须有文本)

**任务定义**: 给定真实数据的首帧作为生成动作的首尾帧（即首尾帧相同），生成循环动画。文本描述循环动作的语义。

**必须有文本**: 首尾帧相同 → 动作语义完全依赖文本。

**设计原理**: 在游戏/动画中，循环动画（idle、walk cycle 等）是核心需求。通过设定 mask[0] = mask[T-1] = 0 且 src_motion[0] = src_motion[T-1] = real_first_frame，模型生成的动作首尾自然衔接，可无限循环。

#### Setting 配置

| Setting ID | 首尾帧来源 | 循环长度 | 说明 |
|-----------|---------|---------|------|
| **T4-A** | GT 首帧 | 60 帧 (2s) | 短循环 |
| **T4-B** | GT 首帧 | 90 帧 (3s) | 中循环 |
| **T4-C** | GT 首帧 | 120 帧 (4s) | 长循环 |

**mask 构建**:
```python
# T4-B: 首尾帧保留（相同的pose）
mask = torch.ones(90, 135)
mask[0, :] = 0      # 首帧保留
mask[-1, :] = 0     # 末帧保留

# src_motion 构建：末帧 = 首帧
src_motion = torch.zeros(90, 135)
first_frame = gt_motion[0]
src_motion[0] = first_frame
src_motion[-1] = first_frame  # 首尾相同
```

**评估指标**:
- **主要**: Loop Continuity Error (首尾帧差异，应接近 0)
- **质量**: Foot Skating, Jitter, Boundary Smoothness
- **语义**: Text-Motion R-Precision, MM-Dist
- **视觉**: 需要人工检查循环播放的流畅度

**使用 datalist**: `eval_loop_animation.json` (200 条, >= 60 帧, 有文本)

---

### 4.5 T5: Motion Prediction

**任务定义**: 给定前缀运动（prefix），预测后续运动。

#### Setting 配置

| Setting ID | Prefix 长度 | 预测长度 | 总序列长度 |
|-----------|-----------|---------|---------|
| **T5-A** | 30 帧 (1s) | 30 帧 (1s) | 60 帧 |
| **T5-B** | 30 帧 (1s) | 90 帧 (3s) | 120 帧 |
| **T5-C** | 60 帧 (2s) | 120 帧 (4s) | 180 帧 |

**mask 构建**:
```python
mask = torch.ones(T, 135)
mask[:prefix_len, :] = 0  # 前缀保留
```

**评估指标**: MPJPE (分时段报告), [P]-MPJPE, Boundary Smoothness, Foot Skating

**使用 datalist**: `eval_transition.json` (500 条)

---

### 4.6 T6: Joint Completion

**任务定义**: 给定部分关节运动，补全其余关节。

#### Setting 配置

| Setting ID | 保留关节 | 生成关节 | 说明 |
|-----------|---------|---------|------|
| **T6-A** | 下半身 (Pelvis, L/R_Hip, L/R_Knee, L/R_Ankle, L/R_Foot, Spine1) + transl | 上半身 | 下→上补全 |
| **T6-B** | 上半身 + transl | 下半身 | 上→下补全 |
| **T6-C** | 左半身 + transl | 右半身 | 左→右镜像 |
| **T6-D** | Root (Pelvis) + Translation | 所有其他关节 | 仅轨迹重建 |
| **T6-E** | 全身除手腕 (20 joints + transl) | L_Wrist + R_Wrist | End-effector |

**评估指标**: MPJPE (生成关节), [P]-MPJPE (保留关节), Jitter

**使用 datalist**: `eval_transition.json` (500 条)

---

### 4.7 T7: Motion Repair

**任务定义**: 检测并修复运动缺陷。

**数据来源**: 直接使用 `motion_annot_web/m2m_database` 管理的低质量数据。数据路径为 `hymotion_m2m_refine_data/data_quality_list/low_quality.json` (85,191 条)。从中采样代表性低质量样本，用修复率作为核心评估指标。

#### Setting 配置

| Setting ID | 模式 | 说明 |
|-----------|------|------|
| **T7-A** | Completion (reactive=0) | 缺陷区域从零重生成 |
| **T7-B** | Editing (reactive=退化值) | 在退化基础上修正 |

**评估指标**:
- **核心**: Repair Success Rate（修复后 MotionQualityChecker 16-checker 通过率）↑
- **辅助**: Over-correction（未标记区域变化 MPJPE）↓
- 对比基线: MoGenDIT ada_denoise 的已有修复记录（16,415 条已通过 / 53,769 已处理 = 30.5% 通过率）

**数据构建**:
- 从 `low_quality.json` 中随机采样 1,000 条低质量样本
- 使用 `quality_eval_manager.py` 的规则引擎自动检测问题帧/关节，生成 checker mask
- M2M 使用 checker mask 进行修复，MoGenDIT 使用 ada_denoise 进行修复
- 统一用 MotionQualityChecker 评估修复后质量

**使用 datalist**: `eval_repair.json` (1,000 条, 从 `low_quality.json` 随机采样, 附带自动检测的 checker 结果)

---

### 4.8 T8: Trajectory-Based Generation (轨迹条件生成，必须有文本)

**任务定义**: 给定文本描述 + 根节点运动轨迹（root translation, 3-dim），生成符合轨迹约束和文本语义的完整动作序列。

**核心目的**: 验证 M2M 的 joint-level mask 能否自然支持 trajectory conditioning（保留 root translation 维度，mask 其余关节），并与 KIMODO 的 trajectory control 能力进行正面对比。

**实现方式**: M2M 通过 joint mask 实现 —— 在 135-dim 表示中，前 3 维为 abs translation，将 mask 的 translation 维度设为 0（保留），其余 132 维设为 1（生成）。这等价于给定轨迹约束下的动作生成。

**对比对象**:
- **KIMODO**: NVIDIA 开源，trajectory control 是其核心能力之一（Phase 2 训练包含 trajectory constraint）。**主要对比对象**。
- **VIVISE**: 腾讯内部，如可获取接口则纳入对比。

#### Setting 配置

| Setting ID | 轨迹来源 | 生成长度 | 文本类型 | 说明 |
|-----------|---------|---------|---------|------|
| **T8-A** | GT root translation | 60 帧 (2s) | simple_caption | 短序列，GT 轨迹 |
| **T8-B** | GT root translation | 120 帧 (4s) | simple_caption | 长序列，GT 轨迹 |
| **T8-C** | GT root translation (稀疏采样, 每10帧) | 120 帧 | simple_caption | 稀疏轨迹约束 |
| **T8-D** | 手工设计轨迹 (直线/圆弧/S形) | 90 帧 (3s) | simple_caption | 人工轨迹，定性对比 |

**mask 构建**:
```python
# T8-A: 保留 root translation，生成其余关节
mask = torch.ones(T, 135)
mask[:, :3] = 0   # 保留 abs translation (前3维)

# T8-C: 稀疏轨迹 — 仅在采样帧保留 translation
mask = torch.ones(T, 135)
traj_indices = list(range(0, T, 10))  # 每10帧
for ti in traj_indices:
    mask[ti, :3] = 0
```

**评估指标**:
- **轨迹保真度**: Trajectory ADE↓ (root 平均位移误差), Trajectory FDE↓ (末帧位移误差), Trajectory Error↓ (保留帧 root L2)
- **动作质量**: MPJPE↓ (非 root 关节), Foot Skating↓, Jitter↓, Quality Pass Rate↑
- **语义匹配**: Text-Motion R-Precision (R@1, R@3), MM-Dist↓
- **重点**: 轨迹保真度 vs 动作自然度的 trade-off

**评测数据**:
- **T8-A/B/C**: 从 `eval_transition.json` (500 条) 中提取 GT root translation 作为轨迹约束
- **T8-D**: 手工设计 20 条典型轨迹（直线、圆弧、S 形、折返），配合 yiran_subset 中的文本

**使用 datalist**: `eval_trajectory.json` (500 条, 从 eval_transition.json 衍生, 提取 root translation) + 手工轨迹 20 条

---

## 5. 测试数据

### 5.1 数据来源

| 属性 | 值 |
|------|-----|
| **数据路径** | `data/hymotion_data/3D/20251111/motions/Game/` |
| **文本路径** | `data/hymotion_data/3D/20251111/improved_simple_caption/Game/` (含 `M_*` 前缀子目录) |
| **质量过滤** | `valid_items_20251225.txt` (20251225 版质量检查通过, 102,121 条 Game) |
| **训练数据日期** | 20251009（训练集 `train_hymotion_400h.json` 中 Game 数据来自 `Game/20251009/`，54,400 条） |
| **测试数据日期** | **20251111**（与训练集不重叠，已代码验证 0 条交集） |
| **数据格式** | NPZ (poses: [T, 156], betas: [1, 16], trans: [T, 3]) |
| **帧率** | 30 fps |

### 5.2 数据不重叠验证

1. **日期隔离**: 训练数据路径 `../hymotion_data/Game/20251009/motions/`，测试路径 `../hymotion_data/3D/20251111/motions/Game/`
2. **代码验证**: `train_hymotion_400h.json` 中 54,400 条 game 样本的 `smplx_path` 均含 `20251009`，0 条含 `20251111`
3. **文件名交叉验证**: 部分文件名重叠（同一游戏的不同版本数据），但这些在 valid_items 质量筛选后已过滤到不同的处理批次

### 5.3 数据规模

| 类别 | 数量 | 说明 |
|------|------|------|
| 全量 Game 动作文件 (20251111) | **111,758** 条 | 88 个 Game 子目录 |
| 质量检查通过 (valid_items_20251225) | **102,121** 条 | Game/ 前缀 |
| 质量通过 + 与训练不重叠 + >= 30 帧 | 大量(未全量扫描) | 基础测试池 |
| **采样进入 master datalist** | **5,430** 条 | 每个 source 最多 200(有caption)+100(无caption) |
| 其中有文本标注 | **985** 条 | 来自 The_Sims 系列为主 |
| 其中无文本标注 | **4,445** 条 | GTA5, 2077, Ark, Witcher, Wuthering Waves 等 |

### 5.4 帧长分布 (master datalist)

| 统计 | 值 |
|------|-----|
| 最短 | 30 帧 (1.0s) |
| 最长 | 359 帧 (12.0s) |
| 均值 | 106.0 帧 (3.5s) |
| 中位数 | 81 帧 (2.7s) |
| >= 60 帧 | 3,427 条 |
| >= 120 帧 | 1,802 条 |
| >= 180 帧 | 941 条 |

### 5.5 主要 Game 来源 (master datalist top 15)

| 来源 | 总数 | 有文本 | 无文本 | 游戏类型 |
|------|------|-------|-------|---------|
| The_Sims_c_MB | 290 | 177 | 113 | 生活模拟 |
| The_Sims_a_MB | 228 | 182 | 46 | 生活模拟 |
| The_Sims_p_MB | 199 | 184 | 15 | 生活模拟 |
| The_Sims_t_MB | 199 | 183 | 16 | 生活模拟 |
| The_Sims_e_MB | 155 | 138 | 17 | 生活模拟 |
| The_Sims_o_MB | 128 | 121 | 7 | 生活模拟 |
| 2077_MB (Cyberpunk 2077) | 100 | 0 | 100 | RPG |
| Ark_PrimalEarth_Female_MB | 100 | 0 | 100 | 生存 |
| Atomic_Heart_P3_MB | 100 | 0 | 100 | FPS |
| GTA5_new_x64c_MB | 100 | 0 | 100 | 开放世界 |
| GTA5_x64c_MB | 100 | 0 | 100 | 开放世界 |
| Hogwarts_Legacy_MB | 99 | 0 | 99 | RPG |
| The_Witcher_dwarf_MB | 98 | 0 | 98 | RPG |
| Wuthering_Waves_468_MB | 98 | 0 | 98 | 动作 |
| The_Witcher_man_MB | 97 | 0 | 97 | RPG |

**文本标注来源**: `improved_simple_caption` 提供 3 级文本:
- `simple_caption`: 一句话简述（用于评测）
- `short_caption`: 2-3 句详述
- `long_caption`: 完整段落描述

> **Note**: 文本标注主要集中在 The_Sims 和 cartwheel 系列。其他游戏来源（GTA5, 2077, Witcher, Hogwarts, Ark, Atomic Heart, Wuthering Waves, Zenless Zone Zero）暂无文本标注，但可用于不需要文本的任务（T1, T2, T5, T6, T7）。

### 5.6 Datalist 文件

所有 datalist 存放于 `data/eval/hymotion_m2m/`：

| 文件 | 数量 | 有文本 | 条件 | 用途 |
|------|------|-------|------|------|
| `eval_datalist_game_20251111.json` | 5,430 | 985 | >= 30 帧, 质量合格 | 主 datalist |
| `eval_t2m.json` | 300 | 300 | 有文本, 文本质量筛选 | **T0 T2M 对比 (游戏数据)** |
| `data/eval/t2m/251125_yiran_subset.json` | 240 | 240 | 人工编写文本 | **T0 T2M 对比 (yiran 文本)** + T0-E 长动作组合 |
| `eval_transition.json` | 500 | 125 | >= 60 帧 | T1 Transition, T5 Prediction, T6 Joint, T8 Trajectory |
| `eval_keyframe.json` | 500 | 139 | >= 120 帧 | T2 Keyframe (需要较长序列) |
| `eval_transition_with_caption.json` | 300 | 300 | >= 60 帧 + 有文本 | T1 有文本版 |
| `eval_keyframe_with_caption.json` | 300 | 300 | >= 120 帧 + 有文本 | T2 有文本版 |
| `eval_first_frame_cond.json` | 300 | 300 | 有文本 (任意长度 >= 30帧) | **T3** 首帧+文本生成 |
| `eval_loop_animation.json` | 200 | 200 | >= 60 帧 + 有文本 | **T4** 循环动画 |
| `eval_repair.json` | 1,000 | — | 低质量样本 (checker 不通过) | **T7** 修复率评估 |
| `eval_trajectory.json` | 500 | 125 | 从 eval_transition 衍生 | **T8** 轨迹条件生成 |

每个 datalist 的 JSON 格式:
```json
{
  "meta": {
    "description": "...",
    "total_items": 500,
    "with_caption": 125,
    "without_caption": 375,
    "sampled_from": 3427
  },
  "data_list": [
    {
      "motion_path": "Game/The_Sims_a_MB/xxx.npz",
      "caption_path": "The_Sims_a_MB/xxx.json",
      "has_caption": true,
      "caption": "A person walks forward slowly.",
      "num_frames": 142,
      "fps": 30,
      "source": "The_Sims_a_MB",
      "duration_sec": 4.73
    }
  ]
}
```

---

## 6. 统一评估流程

### 6.1 Pipeline

```
输入: GT motion (NPZ) + task definition + (optional) text
  │
  ├── 1. 加载 & 表示转换
  │     NPZ → 135-dim (abs transl + rot6d row-major)
  │     via LoadSmplx55(rot_type='rotation_6d', transl_type='abs', smpl_type='smpl_22')
  │
  ├── 2. Mask 构建
  │     按 Task & Setting 生成 (T, 135) mask
  │
  ├── 3. Text 编码 (if applicable)
  │     simple_caption → Qwen3 + CLIP-L → vtxt_input, ctxt_input
  │
  ├── 4. 推理
  │     M2M: HyMotionM2MPipeline(num_steps=50)
  │     竞品: 各自 pipeline
  │
  ├── 5. FK 转换
  │     All outputs → SMPL FK → (T, 22, 3) joint positions
  │
  └── 6. 指标计算
        统一 MotionEvaluator 计算所有指标
```

### 6.2 竞品评估执行

| 竞品 | 评测任务 | 体验方式 | 表示转换 | 推理配置 |
|------|---------|---------|---------|---------|
| HunyuanMotion T2M 1.0 | **T0 (定量+定性)** | 本地推理 + Web Playground | 同表示 (135-dim) 或 SMPL retarget | 内部 pipeline |
| KIMODO | **T0, T1, T2, T3, T6, T8** | 本地 Python API | 135→333-dim (global rot) | DDIM 100 steps, CFG w=2.0 |
| Maya 2026 | T1, T2, T5 | Maya 内置 ML Motion 功能 → 导出 FBX | FBX → retarget SMPL 22-joint | Maya ML Motion 默认参数 |
| VIVISE | T1, T2, T8 (待确认) | 内部接口（需协调） | 待确认 | 待确认 |
| MoGenDIT | T7 | 内部 pipeline | 135→201-dim | 10 steps DDIM |

---

## 7. 结果呈现 (模板)

### Table 0: Text-to-Motion (T0) — T2M 能力退化验证

> **核心问题**: M2M 混合训练（M5 full_mask 仅占 5%）是否导致 T2M 能力显著退化？

#### Table 0a: 游戏数据评测 (eval_t2m.json)

| Method | Setting | FID↓ | R@1↑ | R@3↑ | MM-Dist↓ | Diversity | Skating↓ | Jitter↓ |
|--------|---------|------|------|------|----------|-----------|----------|---------|
| **HunyuanMotion T2M 1.0** | T0-D | — | — | — | — | — | — | — |
| **KIMODO** (T2M mode) | T0-D | — | — | — | — | — | — | — |
| **M2M (Ours, mask=all 1)** | T0-A~D | — | — | — | — | — | — | — |

#### Table 0b: Yiran 测评文本 (251125_yiran_subset.json, 240 条)

| Method | Setting | FID↓ | R@1↑ | R@3↑ | MM-Dist↓ | Diversity | Skating↓ | Jitter↓ |
|--------|---------|------|------|------|----------|-----------|----------|---------|
| **HunyuanMotion T2M 1.0** | T0-D | — | — | — | — | — | — | — |
| **KIMODO** (T2M mode) | T0-D | — | — | — | — | — | — | — |
| **M2M (Ours, mask=all 1)** | T0-A~D | — | — | — | — | — | — | — |

#### Table 0c: 长动作组合生成 (T0-E, yiran_subset 文本组合)

| Method | #Segments | Total Length | Skating↓ | Jitter↓ | Quality Pass Rate↑ | 视觉流畅度 |
|--------|-----------|-------------|----------|---------|-------------------|-----------|
| **HunyuanMotion T2M 1.0** | 2-4 | 240-600f | — | — | — | 定性 |
| **KIMODO** | 2-4 | 240-600f | — | — | — | 定性 |
| **M2M (Ours)** | 2-4 | 240-600f | — | — | — | 定性 |

#### Table 0d: 学术模型论文参考水位 (HumanML3D, 仅参考)

| Method | FID↓ | R@1↑ | R@3↑ | MM-Dist↓ | Diversity |
|--------|------|------|------|----------|-----------|
| MoMask (CVPR'24) | 0.045 | 0.521 | 0.790 | 2.958 | — |
| T2M-GPT (CVPR'23) | 0.116 | 0.491 | 0.775 | 3.007 | 9.761 |
| MLD (CVPR'23) | 0.473 | 0.390 | 0.665 | 3.196 | 9.724 |
| MDM (ICLR'23) | 0.544 | 0.320 | 0.611 | 5.566 | 9.559 |

> **分析重点**:
> - M2M vs HunyuanMotion T2M 1.0 的 FID/R@k 差距 → 混合训练的代价
> - 如果 T2M 退化明显（如 FID 恶化 > 50%），需考虑增大 M5 权重或分阶段训练

### Table 1: Motion Transition (T1)

| Method | Setting | MPJPE↓ | [P]-MPJPE↓ | Bound.Smooth↓ | Skating↓ | Jitter↓ |
|--------|---------|--------|-----------|--------------|---------|---------|
| KIMODO | T1-D | — | — | — | — | — |
| Maya 2026 | T1-D | — | — | — | — | — |
| UMO (paper) | T1-C | 8.55 | 0.95 | — | — | — |
| **M2M (Ours)** | T1-A~F | — | — | — | — | — |

### Table 2: Keyframe Interpolation (T2) — 不同稀疏度

| Method | Interval | #KF/120f | MPJPE↓ | [P]-MPJPE↓ | Traj.Err↓ |
|--------|----------|----------|--------|-----------|-----------|
| KIMODO | 30f | ~4 | — | — | — |
| Maya 2026 | 30f | ~4 | — | — | — |
| **M2M** | 5f~90f | 2~24 | — | — | — |

### Table 3: First-Frame + Text Generation (T3)

| Method | Setting | R@1↑ | R@3↑ | MM-Dist↓ | FID↓ | Skating↓ |
|--------|---------|------|------|----------|------|----------|
| KIMODO | T3-B | — | — | — | — | — |
| **M2M (Ours)** | T3-A~D | — | — | — | — | — |

### Table 4: Loop Animation (T4)

| Method | Length | Loop.Cont.Err↓ | R@3↑ | Skating↓ | Jitter↓ | Bound.Smooth↓ |
|--------|--------|---------------|------|----------|---------|--------------|
| **M2M (Ours)** | 60f | — | — | — | — | — |
| **M2M (Ours)** | 90f | — | — | — | — | — |
| **M2M (Ours)** | 120f | — | — | — | — | — |

### Table 7: Motion Repair (T7) — 低质量数据修复率

> **数据源**: `motion_annot_web/m2m_database` 管理的 `low_quality.json` (85,191 条)

| Method | Mode | #Samples | Repair Success Rate↑ | Over-correction↓ | 备注 |
|--------|------|----------|---------------------|------------------|------|
| MoGenDIT (ada_denoise) | T7-B | 53,769 (已有记录) | 30.5% (16,415/53,769) | — | 已有历史修复数据 |
| **M2M (Ours)** | T7-A | 1,000 | — | — | checker mask, reactive=0 |
| **M2M (Ours)** | T7-B | 1,000 | — | — | checker mask, reactive=退化值 |

### Table 8: Trajectory-Based Generation (T8) — vs KIMODO

> **核心问题**: M2M 的 joint mask 是否可以自然支持 trajectory conditioning，与 KIMODO 的专用 trajectory control 相比如何？

| Method | Setting | Traj.ADE↓ | Traj.FDE↓ | MPJPE↓ | R@3↑ | Skating↓ | Jitter↓ |
|--------|---------|-----------|-----------|--------|------|----------|---------|
| **KIMODO** (trajectory mode) | T8-A | — | — | — | — | — | — |
| **KIMODO** (trajectory mode) | T8-B | — | — | — | — | — | — |
| **M2M (Ours)** | T8-A | — | — | — | — | — | — |
| **M2M (Ours)** | T8-B | — | — | — | — | — | — |
| **M2M (Ours)** | T8-C (稀疏) | — | — | — | — | — | — |

---

## 8. 实施计划

### Phase 1: 数据准备 ✅ (已完成)

- [x] 确认测试数据与训练数据不重叠
- [x] 构建质量过滤后的 datalist（valid_items_20251225）
- [x] 按任务需求构建 7 个 datalist 文件
- [x] 文档化数据来源和统计信息
- [ ] 构建 `eval_repair.json` (从 low_quality.json 采样 1,000 条)
- [ ] 构建 `eval_trajectory.json` (从 eval_transition.json 衍生, 提取 root translation)
- [ ] 构建 T0-E 长动作组合数据 (从 yiran_subset 中组合 2-4 条文本)

### Phase 2: 评测基础设施 (1 周)

| 优先级 | 任务 | 输出 |
|-------|------|------|
| P0 | 实现统一 `MotionEvaluator` 类 | `hftrainer/evaluation/motion_evaluator.py` |
| P0 | 实现各任务 mask 构建工具 | `hftrainer/evaluation/mask_builder.py` |
| P0 | M2M 推理 + 评估的 end-to-end 脚本 | `scripts/eval_m2m.py` |
| P0 | T0 T2M 评测脚本（mask=all 1）| `scripts/eval_m2m_t2m.py` |
| P0 | T8 Trajectory 评测脚本 | `scripts/eval_m2m_trajectory.py` |
| P0 | 构建 `eval_t2m.json` datalist | 从有文本样本中按文本质量筛选 300 条 |
| P0 | 集成 yiran_subset 评测数据 | 解析 `"text#frames#none#id"` 格式 → 标准评测输入 |
| P1 | 文本编码预计算 | 缓存 vtxt/ctxt 到 disk |

### Phase 3: 竞品体验与复现 (1-2 周)

| 优先级 | 任务 | 输出 |
|-------|------|------|
| **P0** | **HunyuanMotion T2M 1.0 本地推理 (T0 定量对比)** | **T2M 1.0 定量数字** |
| P0 | KIMODO 本地部署 + 推理 (T0, T1, T2, T3, T6, **T8**) | KIMODO 定量数字 |
| P0 | **Maya 2026 ML Motion 功能评测 (T1, T2, T5)** | Maya FBX → SMPL retarget → 定量数字 |
| P0 | HunyuanMotion Playground 体验 (T0 定性) | 截图/视频对比 |
| P1 | **VIVISE 内部接口协调 (T1, T2, T8)** | VIVISE 定量数字（如可获取） |
| P1 | MoGenDIT 修复评测 (T7) — 用 low_quality.json 数据 | MoGenDIT 修复率 |

### Phase 4: 我方模型评测 (1 周)

| 优先级 | 任务 |
|-------|------|
| P0 | **M2M T0 (mask=all 1) T2M 评测 — 游戏数据 + yiran 文本 + 长动作组合** |
| P0 | M2M 全量 T1-T7 评测 |
| P0 | **M2M T8 Trajectory 评测 + 与 KIMODO 对比** |
| P0 | T7 修复率评测 — 用 low_quality.json 1,000 条样本 |
| P0 | 文本 vs 无文本消融 |
| P1 | 消融实验（mask 策略、ODE steps、guidance scale）|
| P1 | M5 权重消融（5% vs 10% vs 15%）— 若 T0 退化严重 |

### Phase 5: 报告汇总 (3 天)

| 优先级 | 任务 |
|-------|------|
| P0 | 填入所有结果表格 |
| P0 | 撰写分析结论 |
| P1 | 代表性 case 渲染视频 |

---

## 9. 参考文献

### 商业/闭源竞品
- KIMODO: Rempe et al., "Scaling Controllable Human Motion Generation", NVIDIA, 2026 ([GitHub](https://github.com/nv-tlabs/kimodo))
- Autodesk Maya 2026: ML Motion In-Betweening (内置功能)
- VIVISE: 腾讯内部 MIB 项目（无公开资料）
- UMO: Cong et al., "Unified In-Context Learning Unlocks Motion Foundation Model Priors", arXiv:2603.15975, 2026
- HunyuanMotion 1.0: Tencent, ([Playground](https://hy-motion.ai/playground), [GitHub](https://github.com/Tencent/HunyuanMotion))
- MoGenDIT: 内部扩散修复框架 (chengxuzuo)

### 学术模型论文参考水位 (仅引用数字)
- MoMask: Guo et al., CVPR 2024 ([GitHub](https://github.com/EricGuo5513/momask-codes))
- T2M-GPT: Zhang et al., CVPR 2023 ([GitHub](https://github.com/Mael-zys/T2M-GPT))
- MLD: Chen et al., CVPR 2023 ([GitHub](https://github.com/ChenFengYe/motion-latent-diffusion))
- MDM: Tevet et al., ICLR 2023 ([GitHub](https://github.com/GuyTevet/motion-diffusion-model))
- PackDiT: 2025 ([arXiv](https://arxiv.org/abs/2501.16551))
