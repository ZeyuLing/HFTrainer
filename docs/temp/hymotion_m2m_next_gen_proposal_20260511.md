# HyMotion M2M 下一代方案：条件解耦编排流匹配 (CDO-FM)

**Condition-Decoupled Orchestration Flow Matching**

文档版本: 2.0 | 日期: 2026-05-13 | 状态: 方案设计 (v2.0: 用 STP (语义时空规划) 替换 SEAT——模型先推理文本对应的时空区域与条件覆盖关系，再生成动作；v1.9: 修正 §4.3 输入格式为 MAN+no_inactive 3通道594-dim；修正 CPOS alpha 为 replacement guidance 机制)

---

## 目录

1. [问题诊断](#1-问题诊断)
2. [设计目标与约束](#2-设计目标与约束)
3. [核心方案概述](#3-核心方案概述)
4. [架构设计](#4-架构设计)
5. [训练策略](#5-训练策略)
6. [双 Root 表征方案：SMPL Root vs KIMODO Root](#6-双-root-表征方案smpl-root-vs-kimodo-root) **[v1.7 简化: 两版均 198-dim，仅替换 translation，在线转换]**
7. [Motion Condition 训练采样分析](#7-motion-condition-训练采样分析)
8. [HyMotion M2M 支持的任务枚举](#8-hymotion-m2m-支持的任务枚举)
9. [实验计划（按优先级排序）](#9-实验计划按优先级排序)
10. [评估指标与任务覆盖](#10-评估指标与任务覆盖)
11. [与前沿方法对比及新颖性分析](#11-与前沿方法对比及新颖性分析)
12. [顶会论文定位](#12-顶会论文定位)
13. [风险与备选方案](#13-风险与备选方案)
14. [实施路线图](#14-实施路线图)


---

## 1. 问题诊断

### 1.1 核心症状

当前 HyMotion M2M 带 caption 的模型在 T2M 任务上几乎无法理解 text 输入，输出动作质量甚至比 unconditional 更差。

### 1.2 根因分析

经过对完整代码库的深度审计，发现问题由**实现 bug** 和**架构局限性**共同导致：

#### 1.2.1 实现 Bug（全部已修复）

以下仅列出**对 v2 训练有实际影响**的 bug。B3 (VACE reactive 泄漏) 仅影响 v1 — v2 从设计之初就采用 `vace_condition_mode='no_inactive'`，不存在该问题。B4 (`cond_mask_prob` 默认值) v2 config 显式覆盖，无实际影响。修复详情见附录 A。

| # | Bug | 严重程度 | 影响范围 | 修复日期 | 核心影响 |
|---|-----|---------|---------|---------|---------|
| B1 | Bundle-level Parameter 不训练/不保存/不同步 | P0 | v2 caption (resume 场景) | 03-27 | null embeddings resume 后丢失 |
| B2-ext | null embedding 加载链断裂 (safetensors 不含 bundle params) | P0 | v2 caption (phase/soar configs) | 05-12 | CFG 推理完全失效；训练时 10% unconditional 样本用全零 null embed |
| B5/B6 | Text token OOD + null embedding 分布 | P1 | v2 caption only | 04-20/21 | text encoding 质量下降 |

> **v2 实际影响总结**：
> - **caption_local**: B1 (resume 场景)、B2-ext (phase/soar configs)、B5、B6 有实际影响
> - **uncond_local**: **无 bug 影响**（不用文本、`cond_mask_prob=0.0` 从不触发 null embeddings、VACE `no_inactive` 模式免疫 reactive 泄漏）
> - 其余 bug: B3 仅 v1 (`split_reactive` 模式)；B4 config 显式覆盖无效果

#### 1.2.2 架构局限性

| # | 局限 | 影响 |
|---|------|------|
| A1 | **Text 与 Motion condition 信号竞争** | Motion condition 信息密度远高于 text，模型注意力自然偏向 motion，text 信号被淹没 |
| A2 | **CFG 对 text 的控制力不足** | 当 motion condition 很强（dense mask）时，有无 text 的输出差异极小，CFG 失效 |
| A3 | **训练数据质量** | 15.5% 低质量数据（滑步/抖动/关节跳变）拉低上限 |
| A4 | **无显式物理约束** | 生成结果不保证脚-地面接触的物理合理性 |

#### 1.2.3 滑步问题根因分析（2026-05-12 代码审计）

评测显示 HyMotion M2M 相比 KIMODO 的最大质量差距是**滑步（foot skating）**。经过对 loss 计算、数据加载等环节的完整代码审计，确认**不是实现 bug，而是设计缺陷**：

| # | 缺陷 | 严重度 | 当前状态 | 修复方式 | Phase 0 纳入? |
|---|------|--------|---------|---------|-------------|
| D1 | **FK keypoint loss 已实现但被禁用** (`keypoints3d_weight=0.0`) | P0 | 代码存在，config 关闭 | Config: 设为 10+ | ✅ 纳入 |
| D2 | **Translation 信号占比过低** (10.2% vs KIMODO 40.5%) | P1 | `trans_dim_weight=5.0` 但仅 3/135 维度 | 提高 `trans_dim_weight` 或 KIMODO Root (§6) | ✅ 纳入 (via §6) |
| D4 | **Local rotation 误差沿运动链放大** | P1 | 无 FK 约束时固有问题 | 启用 FK loss (D1) 即可缓解 | ✅ 纳入 (via D1) |

> **已确认不纳入 Phase 0 的方案**:
> - ~~D3: Translation augmentation (`transl_aug_prob`)~~: KIMODO Root 方案使用 ADMM 平滑替代 augmentation，效果更佳且更合理。SMPL Root 版本暂不启用 transl_aug，以保持两版实验的纯对比。
> - ~~D5: Foot contact / ground constraint 监督~~: 需要新增 foot contact 通道（扩展运动表征维度），复杂度高，延后到后续实验。
> - ~~TCC (Typed Condition Canvas)~~: 已从方案中移除（v1.7）。

**代码验证**：translation-body motion 耦合实现正确。`load_smplx.py` 中 `process_transl()` 对 translation 和 root orientation 做一致的旋转增强；loss 计算中 translation (dims 0-3) 和 rotation (dims 3-135) 均参与 velocity loss，无遗漏。

**Phase 0 config-only 修复**（不改代码）：
1. `keypoints3d_weight=10.0`：启用已实现的 FK keypoint loss

#### 1.2.4 Translation 表征对比：HyMotion vs KIMODO

KIMODO 在滑步抑制上显著优于 HyMotion，其**运动表征设计**是关键因素之一。以下对比两者 translation 表征差异，作为实验消融的依据。

| 特性 | HyMotion M2M v2 (198-dim) | KIMODO (369-dim, SOMA-30) |
|------|--------------------------|--------------------------|
| **Translation** | [0:3] 绝对 pelvis XYZ，无平滑 | [0:3] ADMM 平滑后的 pelvis XYZ (margin=6cm) |
| **Heading** | 隐含在 root rotation 中 | [3:5] 显式 `[cos θ, sin θ]` heading angle |
| **Joint positions** | [135:198] 21×3 (XZ rel pelvis, Y abs, pelvis 除外) | [5:95] 30×3 (XZ rel pelvis, Y abs, pelvis 包含) |
| **Rotation** | [3:135] 22×6 rot6d (local frame) | [95:275] 30×6 cont6d (**world frame**) |
| **Velocity** | 隐含 (diffusion 学习 frame delta) | [275:365] 30×3 **显式 joint velocity** |
| **Foot contact** | 无 | [365:369] 4-dim binary (L/R heel/toe) |
| **关节数** | 22 (SMPL) | 30 (SOMA) |
| **Translation augmentation** | 已实现但禁用 (`transl_aug_prob=0.0`) | 无需（ADMM 平滑提供正则化） |

**关键差异分析**：

1. **ADMM 平滑 translation** (KIMODO): 对 root XZ 轨迹施加 ADMM 优化 (margin ≤ 6cm)，去除高频抖动，使 translation 更平滑连续。HyMotion 直接使用原始 MoCap translation，包含噪声抖动，增加学习难度。
2. **显式 velocity 通道** (KIMODO [275:365]): 模型直接看到关节速度信息，有利于学习时间一致性和滑步检测。HyMotion 的 velocity 完全隐含在 flow matching 的帧间差分中。
3. **Foot contact 信号** (KIMODO [365:369]): 二值接触标签 (height < 0.15m ∧ velocity < 0.10 m/s) 提供显式地面约束。HyMotion 完全没有此信息。
4. **World-frame rotation** (KIMODO [95:275]): 使用全局旋转而非局部旋转，避免局部旋转误差沿运动链传播放大 (D4)。代价是 canonical pose 无关性较弱。

**实验消融计划**（纳入 Phase 0）：

| 实验 | 变更 | 预期效果 | 成本 |
|------|------|---------|------|
| **E-T1: ADMM 平滑 translation** | KIMODO Root (§6): 对 [0:3] translation 施加 ADMM 平滑 (margin=6cm) | 减少 translation 高频噪声，改善滑步 | 低（预处理 + 重训） |
| **E-T2: 显式 velocity 通道** | 扩展 198-dim → 261-dim: 增加 21×3 joint velocity [198:261] | 改善时间连续性和滑步检测 | 中（改 representation + 重训），**延后到后续实验** |

> **已移除的实验**：
> - ~~E-T3: Translation augmentation~~: 由 KIMODO Root ADMM 平滑替代
> - ~~E-T4: Foot contact 通道~~: 延后到后续实验 (E7)

### 1.3 核心矛盾

> 当前架构将 text 和 motion 条件混在同一个 conditioning pipeline 中，但两者的**信息密度天然不对称**：一个 dense keyframe condition 包含的约束信息量远大于一句 caption。模型在 multi-task 训练中自然学到"有 motion condition 时忽略 text"的捷径策略（shortcut learning）。

这不是 TAP/TAL 等梯度 trick 能根本解决的问题，需要**架构层面的条件解耦**。

---

## 2. 设计目标与约束

### 2.1 功能需求

| # | 需求 | 优先级 |
|---|------|--------|
| R1 | 支持任意 condition-target pattern（稀疏/稠密、position/rotation、任意关键帧数量/位置） | P0 |
| R2 | 同时理解 motion condition 与 text condition，不互相干扰 | P0 |
| R3 | 兼容 T2M / 带条件 T2M / 语义编辑 / 动作修复 等全部任务 | P0 |
| R4 | 解决滑步等生成质量问题 | P0 |
| R5 | 方案具备顶会论文新颖性 | P1 |
| R6 | 实现可行，不引入过大额外计算量（< 1.5x 当前训练成本） | P2 |

### 2.2 非功能约束

- 沿用 MMDiT (dual-stream → single-stream) 架构骨架，最大化复用已有代码和预训练权重
- 保持 198-dim 运动表征（translation(3) + rotation(132) + position(63)）和 SMPL-22 骨骼，translation 的表达方式可变（raw / ADMM-smoothed）
- 训练规模: 每个实验 48 GPU × V100
- 与现有 eval pipeline (E1-E15) 完全兼容

---

## 3. 核心方案概述

### 3.1 方案名称

**条件解耦编排流匹配 (Condition-Decoupled Orchestration Flow Matching, CDO-FM)**

### 3.2 一句话描述

> 通过**语义时空规划 (STP)**——让模型先推理"文本内容对应动作的什么时间、什么部位，哪些已被 motion condition 给定，哪些需要从文本规划"，再执行实际生成——从根本上解决 text/motion condition 信息密度不对称导致的 shortcut learning。结合渐进密度课程训练 (PDCT)、条件渐进 ODE 采样 (CPOS) 和 KIMODO 风格 smooth root trajectory，使单一模型在纯文本生成与强条件补全之间平滑过渡。

### 3.3 核心创新点

1. **Progressive Density Curriculum Training (PDCT)**: 渐进密度课程训练。核心洞察：text 和 motion condition 的信息密度不对称是 shortcut learning 的根因，但如果在训练初期强制模型**只看到低密度 motion condition**（甚至纯 T2M），text pathway 就必须先被建立；后期逐步引入高密度条件，模型已经建立了 text 理解能力，不会再退化为忽略 text 的 shortcut。这与人类学习先学概念再学细节的认知规律一致。**零额外参数，可直接从 E1-E4 checkpoint 继续训练。**

2. **Dual Root Representation + Loss Alignment**: 两套 root 表征方案 (SMPL raw / KIMODO smooth trajectory) 的 A/B 实验，统一 position loss 在 relative-to-root 空间计算，移除 t² weighting

3. **Unified V3 Condition Sampler**: Rank-K Boolean Tensor Prior，统一 caption/uncond configs 的 motion condition 训练采样

4. **Semantic-Temporal Planning (STP)**: 语义时空规划。核心洞察：text/motion condition 竞争的根因不在于 dropout 概率或数据构造，而在于**模型缺乏对"文本语义 → 时空区域"映射关系的显式推理能力**。给定 text "一个人先走几步然后坐下" 和 motion condition（前2秒 keyframe），模型应当先理解：(a) "走几步"对应前半段、全身运动，已被条件覆盖 → 从 condition 复用；(b) "坐下"对应后半段、核心躯干 + 下肢，未被条件覆盖 → 需从 text 规划。STP 通过在生成流程中引入**语义-时空接地 (Semantic-Temporal Grounding)** 阶段，让模型显式产出文本各语义片段到 (时间段, 身体部位, 条件覆盖状态) 的映射，再以此指导后续生成。**与任务类型无关 (T2M/M2M/inpainting/editing 统一适用)**。

5. **Condition-Progressive ODE Sampling (CPOS)**: 条件渐进 ODE 采样。核心洞察：flow matching 的 ODE 求解从 t=0（纯噪声）到 t=1（干净数据），早期 step 确定全局结构/语义，后期 step 精细化局部细节。Text 是全局语义信号，motion condition 是局部结构约束——二者的信息粒度天然对应 ODE 的不同阶段。CPOS 在 ODE 前期增强 text 的 CFG 权重、减弱 motion condition 约束；后期反转。**零额外参数，仅修改推理时的 CFG schedule，可直接用于任何已训练模型。**

---

## 4. 架构设计

### 4.1 整体架构

```
                         ┌─────────────────────────────────────┐
                         │          CDO-FM Architecture         │
                         └─────────────────────────────────────┘

  Text Input                                          Motion Condition Input
  "a person walks                                     [VACE no_inactive: x_t(MAN) + reactive + mask]
   forward and waves"                                 [sparse keyframes / trajectory / ...]
       │                                                      │
       ▼                                                      ▼
  ┌─────────────┐                                    ┌──────────────────┐
  │ Text Encoder │ (frozen Qwen3 + CLIP-L)           │ VACE Condition   │
  │             │                                    │ Encoder (现有)    │
  └──────┬──────┘                                    └────────┬─────────┘
         │                                           cond (B,L,198)
   ctxt (B,S,4096)                                   mask (B,L,198)
   vtxt (B,1,768)                                    density ρ = 1 - mask.mean()
         │                                                  │
         │           ┌────────────────────────────┐         │
         │           │  Training: PDCT Schedule    │         │
         │           │  (condition density ramp)   │         │
         │           │  + STP (semantic-temporal    │         │
         │           │    planning)                 │         │
         │           └────────────────────────────┘         │
         │                                                  │
         ▼                                                  ▼
  ╔══════════════════════════════════════════════════════════╗
  ║            MMDiT Backbone (reused, no new params)        ║
  ║                                                          ║
  ║  ┌──────────────────────────────────────────────┐       ║
  ║  │  Dual-Stream Blocks (×N_double)              │       ║
  ║  │    Motion Stream + Text Stream               │       ║
  ║  │    Joint Attention (unchanged)               │       ║
  ║  └──────────────────────────────────────────────┘       ║
  ║                                                          ║
  ║  ┌──────────────────────────────────────────────┐       ║
  ║  │  Single-Stream Blocks (×N_single)            │       ║
  ║  │    [motion_tokens; text_tokens] → attn → FFN │       ║
  ║  └──────────────────────────────────────────────┘       ║
  ║                                                          ║
  ╚══════════════════════════════════════════════════════════╝
         │
         ▼
  ┌─────────────────┐
  │  Flow Velocity  │
  │  Prediction     │
  │  v_θ (B,L,198)  │
  └─────────────────┘
         │
         ▼
  ┌──────────────────┐
  │  CPOS ODE Solve  │
  │  (text-heavy →   │
  │   motion-heavy   │
  │   CFG schedule)  │
  └──────────────────┘
         │
         ▼
    Output Motion (B, L, 198)
```

### 4.2 模块详细设计

#### 4.2.1 Progressive Density Curriculum Training (PDCT)

**动机**: Text 和 motion condition 的信息密度天然不对称（§1.3）。当两者同时呈现给模型时，模型倾向于忽略信息密度低的 text（shortcut learning）。现有方法（如 DensityModulator 等门控机制）试图通过引入可学习的参数来平衡两者，但：(a) 引入了新参数，无法复用已有 checkpoint；(b) 门控本身也可能学到 shortcut（总是关闭 text gate）。

**核心洞察**: 类比人类学习——先学"什么是走路"（概念/语义），再学"走路的具体姿势"（结构/约束）。如果在训练初期**强制限制 motion condition 密度**，模型必须依赖 text 来理解任务语义，从而**先建立 text pathway**。一旦 text 理解能力被建立，后续引入高密度 motion condition 时，模型已经学会了从 text 提取有用信息的能力，不会退化为 shortcut。

**设计**: 三阶段课程学习，**零额外参数**，仅修改 V3 Condition Sampler 的采样分布调度。

```
Phase A — Text Foundation (步数 0 ~ S_A):
  目标: 建立 text understanding pathway
  V3 Sampler 配置:
    K 分布: πK = (0.40, 0.50, 0.10, 0.00, 0.00)  — 40% 纯 T2M (K=0), 50% 单 atom, 无高阶
    时间 primitive 偏置: empty(0.4) + periodic-大步长(0.3) + interval-短(0.2) + renewal(0.1)
    → 条件密度期望值 E[ρ] ≈ 0.15 (很稀疏, text 是主要语义来源)
  
Phase B — Density Ramp (步数 S_A ~ S_B):
  目标: 逐步引入高密度条件，同时保持 text pathway
  V3 Sampler 配置随步数线性插值:
    πK: 从 Phase A 分布线性过渡到目标分布 (0.10, 0.55, 0.25, 0.07, 0.03)
    条件密度期望值: E[ρ] 从 0.15 线性增长到 0.55
  关键: 过渡速度不能太快，否则 text pathway 退化
  
Phase C — Full Distribution (步数 > S_B):
  目标: 正常训练，覆盖全部条件密度分布
  V3 Sampler 配置: 标准分布 πK = (0.10, 0.55, 0.25, 0.07, 0.03)
  此时模型已具备 text 理解能力，高密度条件不会导致 shortcut
```

**从 E1-E4 checkpoint 继续训练的策略**:

```
E1/E3 (uncond) 的 checkpoint → 直接用作 Phase A 起点
  原因: uncond 模型已经学习了 motion 生成的基本能力
  从 Phase B 开始（跳过 Phase A），因为 uncond 没有 text pathway

E2/E4 (caption) 的 checkpoint → 从 Phase B 中期开始
  原因: caption 模型已有初步 text 理解（尽管 shortcut 存在）
  用低密度条件分布 "回退" 一段时间，强化 text pathway
  然后正常进入 Phase C
```

**参数选择指南**:
- `S_A`: 建议 5K-10K steps（Phase A 不需要太长，只需 text pathway 初步建立）
- `S_B`: 建议 S_A + 15K-25K steps（Ramp 阶段需要足够长以避免 text pathway 退化）
- 从 checkpoint 继续时，S_A 可跳过，S_B 缩短到 10K-15K steps

**理论支撑**:
1. **Curriculum Learning** (Bengio et al., 2009): 从简单到复杂的样本排序可以加速收敛并改善泛化
2. **Information Bottleneck**: Phase A 中 text 是 bottleneck——模型被迫通过 text bottleneck 编码语义信息
3. **Anti-Shortcut Regularization**: 通过控制训练分布序列，破坏 "有 motion condition → 忽略 text" 的因果链

### 4.3 输入张量格式

v2 采用 **MAN (Mask-Aware Noise) + `no_inactive` 模式**，不使用 inactive 通道——已知位置的条件值直接替换进 x_t:

**训练时 (MAN)**:
```python
# hymotion_m2m_trainer.py: mask_aware_noise=True
keep_mask = 1 - src_mask        # (B, L, D): 1=已知位置, 0=待生成位置
x_t = x_t * src_mask + x1 * keep_mask
# 已知位置 (mask=0, keep=1): x_t = x1 (干净条件值, 不加噪)
# 待生成位置 (mask=1, keep=0): x_t = 原始噪声混合
```

**推理时 (Replacement Guidance)**:
```python
# hymotion_m2m_pipeline.py: use_replacement=True, rep_mode='skip_last'
if not is_last_step:
    x = torch.where(keep_mask, x_clean, x)  # 每步替换已知位置为干净值
```

**3 通道输入格式** (`vace_condition_mode='no_inactive'`):

| 通道 | 维度 | 含义 |
|------|------|------|
| x_t (MAN) | (B,L,198) | 噪声状态，已知位置已被替换为 x1 干净值 |
| reactive | (B,L,198) | `src_motion * src_mask`，待生成区域的反应通道 |
| mask | (B,L,198) | 1=待生成, 0=已知条件 |
| **模型输入** | **(B,L,594)** → proj | **3×D，不使用 inactive 通道** |

关键设计: MAN 将条件信息直接编码进 x_t（已知位置 = 干净值），模型从 x_t 自身读取条件信息，无需额外的 inactive 通道。CDO-FM 的 PDCT/STP/CPOS 均在训练调度和推理 schedule 层面操作，不改变此 3 通道输入格式。

### 4.4 Condition-Progressive ODE Sampling (CPOS)

**动机**: 标准 CFG 对 text 和 motion condition 施加相同的引导强度，且在 ODE 所有步骤中保持不变。但 flow matching 的 ODE 从 t=0（纯噪声）到 t=1（干净数据）的不同阶段承担不同的生成任务：早期确定全局结构和语义，后期精细化局部细节和空间约束。

**核心洞察**: Text 是**全局语义**信号（"走路"、"高兴地跳"），motion condition 是**局部结构**约束（"帧 30 右手在位置 X"）。这种信息粒度的差异天然对应 ODE 的不同阶段：
- **ODE 早期 (t→0)**: 高噪声，模型确定"做什么动作"→ text 信息最有价值
- **ODE 后期 (t→1)**: 低噪声，模型精细化"怎么做这个动作"→ motion condition 约束最关键

CPOS 利用这一洞察，设计**时间自适应的 CFG schedule**。

**设计**: 零额外参数，仅在推理时修改 CFG 权重随 ODE timestep 的变化 schedule。

```python
def cpos_cfg_schedule(t: float, w_text_base: float = 7.5, w_motion_base: float = 1.0,
                       text_peak: float = 0.3, motion_onset: float = 0.4) -> Tuple[float, float]:
    """
    Condition-Progressive ODE Sampling schedule.
    
    在 ODE 早期增强 text CFG，后期增强 motion condition 约束。
    使用标准 CFG (单次 conditional + 单次 unconditional = 2 forward passes)，
    通过调制 CFG scale 实现条件渐进——不需要额外的 forward pass。
    
    Args:
        t: ODE timestep, 0=noise, 1=clean
        w_text_base: text CFG 基础权重
        w_motion_base: motion condition 引导基础权重
        text_peak: text CFG 峰值位置 (t 值)
        motion_onset: motion condition 引导开始增强的位置
    
    Returns:
        w_text(t): text CFG 权重 at timestep t
        alpha(t): motion condition 混合系数 (0=不使用条件, 1=完全条件)
    """
    # Text CFG: bell-shaped, 在 ODE 早期到中期达到峰值
    # 直觉: 早期需要语义引导确定动作类型，中期仍需语义保持一致性，后期衰减
    w_text = w_text_base * exp(-((t - text_peak) / 0.25)**2)
    
    # Motion condition: 通过 replacement guidance 强度控制
    # v2 推理使用 replacement guidance: 每步将已知位置替换为干净值
    # alpha 控制替换强度:
    #   alpha=0: 不执行替换 → 已知位置也被自由生成 → 等同于无条件
    #   alpha=1: 完全替换 → 已知位置保持精确 → 完全约束
    # 实现: x_known = alpha * x_clean + (1-alpha) * x_denoised  (在条件帧处)
    alpha = sigmoid(10 * (t - motion_onset))
    
    return w_text, alpha


def cpos_ode_step(model, x_t, t, text_emb, null_text_emb, motion_cond, mask, 
                   w_text_base=7.5, **kwargs):
    """
    CPOS 的单步 ODE 求解。
    仍然只需 2 次 forward pass (标准 CFG)，不增加推理开销。
    """
    w_text, alpha = cpos_cfg_schedule(t, w_text_base)
    
    # 条件渐进: replacement guidance 强度随 ODE 进程逐渐增强
    # 早期: alpha≈0, 不替换已知位置 → 模型必须依赖 text 确定语义
    # 后期: alpha≈1, 完全替换已知位置 → 精确空间约束
    # (区别于 v2 现有的 skip_last 模式: CPOS 使 alpha 从 0 渐变到 1，而非 0/1 开关)
    
    # Standard CFG with time-varying text weight
    v_cond = model(x_t, text_emb, motion_cond, mask)
    v_uncond = model(x_t, null_text_emb, motion_cond, mask)
    v_guided = v_uncond + w_text * (v_cond - v_uncond)
    
    # ODE step → 得到 x_{t+dt}
    x_next = x_t + v_guided * dt
    
    # Replacement guidance with progressive alpha
    # 在条件帧处: x_next = alpha * x_clean + (1-alpha) * x_next
    if alpha > 0 and not is_last_step:
        x_next = torch.where(keep_mask, alpha * x_clean + (1-alpha) * x_next, x_next)
    
    return x_next
```

**关键设计决策**:
- **仅 2 次 forward pass**: 与标准 CFG 相同的推理开销，不像 Decoupled CFG 需要 3-4 次
- **Text CFG bell-shaped schedule**: 不是简单的线性衰减，而是在 ODE 早期-中期保持高 text 引导（确保语义一致性），最后才衰减
- **Motion condition progressive reveal**: 通过 alpha 控制 replacement guidance 强度，已知位置从"不替换"渐变为"完全替换"，避免早期 motion condition 约束淹没 text 语义信号
- **可调参数**: `text_peak`, `motion_onset` 控制两者的时间分配，不同任务（T2M vs dense M2M）可以用不同的 schedule

**与 Decoupled CFG 的对比**:

| 特性 | Decoupled CFG (已废弃) | CPOS |
|------|----------------------|------|
| Forward passes per step | 3-4 | **2** (与标准 CFG 相同) |
| 额外参数 | 无 | **无** |
| 训练修改 | 需要 4 种 dropout 组合 | **无需修改训练** |
| Checkpoint 兼容性 | 需要重新训练 | **直接用于任何已有模型** |
| 理论基础 | Ad hoc 分解 | **ODE 语义-结构时间分离** |

**理论支撑**:
1. **Diffusion/Flow ODE 的粗-细粒度特性**: 已被广泛验证——ODE 早期决定全局结构，后期精细化细节 (Progressive Distillation, Meng et al. 2023)
2. **条件引导的时间依赖性**: Imagen (Saharia et al., 2022) 发现 dynamic thresholding 在不同 ODE 阶段应采用不同策略
3. **信息论视角**: 在高噪声时，motion condition 的 mutual information 与 target 较低（因为条件值本身被噪声淹没），text 的 mutual information 相对更高（全局语义不受空间噪声影响）

---

## 5. 训练策略

### 5.1 Semantic-Temporal Planning (STP) — 语义时空规划

**动机**: 信息密度不对称导致 shortcut learning 的根因不在于数据构造或 dropout 策略——即使混入语义编辑数据让 text 在特定样本上不可替代，模型仍然缺乏**通用的文本-时空推理能力**：它不知道文本中的每个语义片段对应动作的哪个时间段、哪些身体部位，也不知道哪些部分已被 motion condition 覆盖、哪些需要从 text 全新规划。这个推理能力与具体任务类型（T2M/M2M/inpainting/editing）无关——任何涉及 text + condition 的生成都需要模型先"想清楚"再"动手做"。

**核心洞察**: 人类动画师拿到一个 text prompt 和部分关键帧约束时，不会直接开始画中间帧——而是先做规划:
1. 解析文本: "先走几步然后坐下" → [走路: 0-3s, 全身] + [坐下: 3-5s, 躯干+下肢]
2. 对照条件: keyframe 在 t=0,1,2s 覆盖了走路阶段 → 走路部分主要从条件推断
3. 识别缺口: 坐下阶段没有条件 → 需要从 text 完全规划姿态序列
4. 执行生成: 走路部分保持条件一致性，坐下部分按 text 语义创造

STP 将这一推理过程**显式建模**到生成流程中。

**设计**: STP 包含三个互补组件，从训练数据、模型推理到生成过程全链路引入语义-时空推理。

#### 5.1.1 Grounded Text Augmentation (GTA) — 接地文本增强

在训练时，将 caption 增强为带有**时空接地标注**的格式，让模型学习文本语义与运动时空区域的对应关系。

```python
# 标准 caption (当前):
"A person walks forward, then sits down on a chair"

# GTA 接地增强后的 caption:
"A person walks forward [T:0.0-0.6, B:full_body], then sits down on a chair [T:0.6-1.0, B:torso+lower]"

# 其中:
# [T:start-end] = 归一化时间段 (0.0=开始, 1.0=结束)
# [B:body_parts] = 涉及的身体部位组 (full_body / upper / lower / torso / arms / legs / head)
```

**实现方式**: 利用 LLM (Qwen3/GPT-4) 对现有 caption 进行时空接地标注，生成 `grounded_caption`。模型训练时以一定概率 (如 50%) 使用 grounded caption 替代原始 caption。

**效果**: 模型通过 cross-attention 学习到 text token 与 motion 时空区域的对齐关系——"walks forward" 的 text token 会自然地 attend to 前半段全身运动区域，"sits down" attend to 后半段下半身区域。这种对齐让模型在推理时能**隐式推断**文本各部分应该影响哪些时空区域。

```python
# GTA 数据预处理
def augment_caption_with_grounding(caption: str, motion_duration: float) -> str:
    """
    用 LLM 将描述性 caption 增强为时空接地格式。
    
    输入: "A person walks forward, then sits down on a chair"
    输出: "A person walks forward [T:0.0-0.6, B:full_body], then sits down on a chair [T:0.6-1.0, B:torso+lower]"
    """
    prompt = f"""Given a motion caption and duration {motion_duration:.1f}s, add temporal and body part annotations.
    Rules:
    - [T:start-end] uses normalized time (0.0 to 1.0)
    - [B:parts] uses: full_body, upper, lower, torso, arms, legs, head
    - Place annotations after each semantic action phrase
    Caption: "{caption}"
    """
    return llm_annotate(prompt)


# 训练时条件构造
def construct_stp_training_sample(motion, caption, grounded_caption, condition_mask):
    """
    以 gta_prob 概率使用接地增强的 caption。
    """
    if random.random() < gta_prob:  # e.g., 0.5
        text = grounded_caption  # 带时空标注
    else:
        text = caption           # 原始描述
    
    return motion, text, condition_mask
```

#### 5.1.2 Condition-Aware Planning Tokens (CAPT) — 条件感知规划标记

在模型架构中引入**轻量规划头 (planning head)**，让模型在生成 flow velocity 之前，先预测一个**语义-时空规划图 (Semantic-Temporal Plan)**——标注每个 (time_step, body_part) 区域的信息来源应该是 text、condition 还是两者混合。

```python
# Planning head: 从 MMDiT backbone 的中间表示产出 planning map
# 输入: backbone 的某层 hidden state h (B, L, D)
# 输出: plan_map (B, L, K) — 每个 timestep 对 K 个身体部位组的来源分配

class PlanningHead(nn.Module):
    """
    轻量规划头。输出每个 (时间步, 身体部位组) 的信息来源权重。
    
    参数量极少: 仅一个线性层 D → K (K=5~7 身体部位组)。
    """
    def __init__(self, hidden_dim: int, n_body_groups: int = 5):
        super().__init__()
        # 5 body groups: torso, left_arm, right_arm, left_leg, right_leg
        # 输出: 每个 group 的 source_weight ∈ [0,1]
        #   0 = 完全从 text 规划, 1 = 完全从 condition 读取
        self.proj = nn.Linear(hidden_dim, n_body_groups)
    
    def forward(self, h, condition_mask):
        """
        h: (B, L, D) — backbone hidden state
        condition_mask: (B, L, 198) — 0=已知, 1=待生成
        
        Returns:
            plan_logits: (B, L, K) — 每个身体部位组的 condition reliance
        """
        plan_logits = self.proj(h)  # (B, L, K)
        return plan_logits  # sigmoid 后: 0=text规划, 1=condition读取


# 训练目标: plan_logits 的 GT 由 condition_mask 生成
def compute_plan_gt(condition_mask):
    """
    从 condition_mask 生成 planning GT。
    
    condition_mask: (B, L, 198) — 0=已知条件, 1=待生成
    
    已知区域 → plan GT = 1 (从 condition 读取)
    待生成区域 → plan GT = 0 (从 text 规划)
    """
    # 将 198-dim mask 映射到 K 个身体部位组
    body_groups = {
        'torso': list(range(0, 3)) + list(range(3, 3+6)),     # trans + root rot
        'left_arm': list(range(3+6*4, 3+6*8)),                 # 4 left arm joints
        'right_arm': list(range(3+6*8, 3+6*12)),               # 4 right arm joints
        'left_leg': list(range(3+6*1, 3+6*4)),                 # 3 left leg joints
        'right_leg': list(range(3+6*12, 3+6*15)),              # 3 right leg joints
    }
    plan_gt = []
    for group_dims in body_groups.values():
        # 该组的条件覆盖率: 0=全部已知(plan=1), 1=全部待生成(plan=0)
        group_mask = condition_mask[..., group_dims].mean(dim=-1)  # (B, L)
        plan_gt.append(1.0 - group_mask)  # 反转: 高=从condition, 低=从text
    plan_gt = torch.stack(plan_gt, dim=-1)  # (B, L, K)
    return plan_gt
```

**关键设计**: CAPT 的 planning head **极轻量** (一个线性层，参数量 < 0.01% 模型总参数)，不破坏"零额外参数"的设计原则。其作用不是"控制生成"，而是作为**辅助训练信号**——迫使 backbone 的内部表示中必须编码文本-条件-时空的三角关系，从而提升 backbone 对 text 信号的利用能力。

**推理时**: planning head 的输出可用于 (a) 可视化模型的"规划图"以增强可解释性，(b) 作为 CPOS 的动态 schedule 输入——在模型认为"需要从 text 规划"的区域加强 text CFG 权重。

#### 5.1.3 Plan-Guided Generation (PGG) — 规划引导生成

将 CAPT 的规划结果与 CPOS 的推理 schedule 联动，实现**自适应的条件引导**。

```python
# 推理时: PGG 将 planning map 反馈到 CPOS schedule
def plan_guided_cpos_step(x_t, v_model, plan_map, t, dt, ...):
    """
    CPOS ODE step with plan-guided adaptive CFG.
    
    plan_map: (B, L, K) — 模型预测的规划图
        高值 = 该区域由 condition 主导 → 降低 text CFG, 增强 replacement
        低值 = 该区域由 text 主导 → 增强 text CFG, 减弱 replacement
    """
    # 基础 CPOS schedule (同 §4.4)
    w_text_base = bell_shaped_cfg(t, text_peak=0.3)
    alpha_base = sigmoid_alpha(t, motion_onset=0.5)
    
    # Plan-guided 调制: 在 text-dominant 区域增强 text CFG
    text_reliance = 1.0 - plan_map.mean(dim=-1, keepdim=True)  # (B, L, 1)
    # text_reliance 高 → 该区域需要更多 text 引导
    w_text = w_text_base * (1.0 + text_reliance * adaptive_scale)
    
    # Plan-guided replacement: 在 condition-dominant 区域增强替换
    cond_reliance = plan_map.mean(dim=-1, keepdim=True)
    alpha = alpha_base * (1.0 + cond_reliance * 0.5)  # 条件区域替换更强
    alpha = alpha.clamp(0, 1)
    
    # 标准 CFG + replacement (同 CPOS)
    v_uncond = v_model(x_t, text=None)
    v_cond = v_model(x_t, text=text_emb)
    v_guided = v_uncond + w_text * (v_cond - v_uncond)
    x_next = x_t + v_guided * dt
    
    if alpha.max() > 0:
        x_next = torch.where(keep_mask, alpha * x_clean + (1-alpha) * x_next, x_next)
    
    return x_next
```

**STP 与 PDCT/CPOS 的协同关系**:

| 组件 | 作用层面 | 解决什么 | 与 STP 的关系 |
|------|---------|---------|-------------|
| PDCT | 训练**时间轴** | 条件密度分布编排——先低后高建立 text pathway | STP-GTA 的接地 caption 在所有 Phase 使用 |
| STP-GTA | 训练**数据** | 文本语义到时空区域的显式对齐 | 为 CAPT 提供训练信号基础 |
| STP-CAPT | 模型**表示** | 迫使 backbone 编码文本-条件-时空三角关系 | 产出规划图，指导 PGG |
| STP-PGG | 推理**过程** | 自适应条件引导——text/condition 按区域调度 | 消费 CAPT 的规划图 + CPOS 的时间 schedule |
| CPOS | 推理**schedule** | ODE timestep 维度的 text/motion 引导调度 | PGG 在 CPOS 基础上叠加空间维度调度 |

**与任务类型的关系**: STP 的设计**完全与任务类型无关**。无论是 T2M (condition=∅, plan 全为 text-dominant)、dense M2M (大部分 plan 为 condition-dominant, 仅空隙从 text 规划)、还是语义编辑 (condition=source, text=编辑方向, plan 需区分保持 vs 修改区域)，STP 都以统一的推理机制工作——模型不需要知道"这是什么任务"，只需推理"文本的哪些部分对应哪些时空区域、与条件的覆盖关系如何"。

### 5.2 数据策略

#### 5.2.1 质量过滤

从 549K 样本切换到 456K 高质量子集（`high_quality.json`），预期 +3-5% 质量提升。

#### 5.2.2 Text Augmentation

当前每条 motion 只有一个 caption。为了增强 text 理解的鲁棒性：
- 同一 motion 的多种描述（不同详细程度、不同重点）
- 通过 LLM 改写现有 caption（已有 `--use-rewritten` 支持）

---

## 6. 双 Root 表征方案：SMPL Root vs KIMODO Root

### 6.1 动机

KIMODO 在滑步抑制上显著优于 HyMotion，根因之一是其 **ADMM smooth root trajectory** 表征。为了验证这一假设并找到最优方案，我们实现两套 root 表征并做 A/B 对比实验。

**关键简化（v1.7 修正）**: 经分析 KIMODO 推理代码，发现 heading channel 在推理时完全未使用，trans_residual 分解也不必要。因此我们仅借鉴 KIMODO 的 **ADMM smooth trajectory** 思想，直接替换 translation [0:3]，**不引入 heading、不引入 trans_residual、不改变维度**。两个版本均为 198-dim。

### 6.2 两版 Root 表征定义

#### 6.2.1 版本 A: SMPL Root（当前实现，baseline）

```
198-dim layout:
  [0:3]     = raw pelvis translation (原始 MoCap world XYZ，含高频噪声)
  [3:9]     = pelvis rotation (rot6d, world-frame)
  [9:135]   = 21 body joint rotations (rot6d, parent-relative)
  [135:198] = 21 × 3 joint positions (XZ relative to raw pelvis, Y absolute)
```

**特点**：
- translation 为原始 MoCap 数据，包含高频噪声/抖动
- 与 SMPL forward kinematics 直接兼容，无需额外转换
- 总维度: 198

#### 6.2.2 版本 B: KIMODO Root（smooth trajectory 替换）

```
198-dim layout (与版本 A 维度完全相同):
  [0:3]     = ADMM-smoothed pelvis translation (smooth XZ, raw Y)
  [3:9]     = pelvis rotation (rot6d, world-frame) — 与版本 A 完全相同
  [9:135]   = 21 body joint rotations (rot6d, parent-relative) — 与版本 A 完全相同
  [135:198] = 21 × 3 joint positions (XZ relative to smooth root, Y absolute)
```

**与版本 A 的唯一区别**:
1. **[0:3] translation**: raw → ADMM-smoothed（XZ 平面 ADMM 优化，margin ≤ 6cm；Y 轴保持不变）
2. **[135:198] position channels 参考系**: relative to raw pelvis → relative to smooth root

**不引入的内容**:
- ~~heading channel~~: KIMODO 推理时未使用，不引入
- ~~trans_residual~~: 不拆分 smooth + residual，不增加维度
- ~~维度变化~~: 两版均为 198-dim，架构代码完全不变

### 6.3 SMPL Trans → KIMODO Trans 在线转换

转换在数据加载时**在线完成**，不需要离线预处理 motion 文件。仅需预先计算版本 B 的 mean/std 统计量。

```python
def smpl_trans_to_smooth_trans(motion_198: Tensor, admm_margin: float = 0.06) -> Tensor:
    """
    在线将 SMPL Root 198-dim 转换为 KIMODO Root 198-dim。
    在 dataset __getitem__ 中调用。
    
    输入/输出: (T, 198)，维度不变
    
    转换:
    1. [0:3] raw_trans → ADMM smooth_trans (XZ平滑, Y不变)
    2. [3:135] rotation 部分完全不变
    3. [135:198] position 参考系从 raw pelvis → smooth root
    """
    raw_trans = motion_198[..., 0:3]           # (T, 3)
    rotation = motion_198[..., 3:135]          # (T, 132) — 透传
    pos_rel_raw = motion_198[..., 135:198]     # (T, 63) = 21×3
    
    # Step 1: ADMM 平滑 translation XZ (Y 保持)
    smooth_trans = admm_smooth_xz(raw_trans, margin=admm_margin)
    
    # Step 2: 调整 position 参考系
    # pos_rel_raw[j] = world_pos[j] - raw_trans (对 XZ)
    # pos_rel_smooth[j] = world_pos[j] - smooth_trans (对 XZ)
    # → pos_rel_smooth[j] = pos_rel_raw[j] + (raw_trans - smooth_trans)
    trans_diff = (raw_trans - smooth_trans)  # (T, 3)
    # 将 trans_diff 广播到 21 个 joint
    trans_diff_expanded = trans_diff.unsqueeze(-2).expand(..., 21, 3).reshape(..., 63)
    pos_rel_smooth = pos_rel_raw + trans_diff_expanded  # 仅 XZ 变化，Y 不变（因为 Y 轴 smooth_trans.y == raw_trans.y）
    
    return torch.cat([smooth_trans, rotation, pos_rel_smooth], dim=-1)  # (T, 198)


def smooth_trans_to_smpl_trans(motion_198_smooth: Tensor, raw_trans: Tensor) -> Tensor:
    """
    推理后将 KIMODO Root 输出转换回 SMPL Root。
    
    注意: 推理输出的 [0:3] 就是 smooth_trans，可直接用于 SMPL 的 translation。
    因为 ADMM smooth_trans 是 raw_trans 的平滑近似（margin ≤ 6cm），
    直接使用 smooth_trans 作为 SMPL translation 即可，误差在 6cm 以内。
    
    如果需要更精确的还原，可以在推理时用原始 translation 做 replace guidance。
    """
    # 直接使用 smooth_trans 作为 SMPL translation（近似，误差 ≤ 6cm）
    return motion_198_smooth  # translation 直接用，rotation 不变
```

**数据预处理仅需**:
```
预计算 KIMODO Root 版本的 mean/std 统计量:
  1. 对全部训练数据在线转换 (smpl_trans_to_smooth_trans)
  2. 计算 198-dim 的新 mean/std
  3. 保存为 mean_std_198dim_kimodo_root.npz

预处理耗时: < 30min (遍历一遍训练集即可)
```

### 6.4 Loss 对齐

#### 6.4.1 Position Loss 在 Relative-to-Root 空间计算

当前实现中 keypoint3d loss 已经在 relative to root 空间计算（参见 `m2m_loss.py:222`）:
```python
local_keypoints3d = pred_keypoints3d[:, :, 1:22] - pred_keypoints3d[:, :, 0:1, :]
```

但 position channels 的 loss（x1 loss、velocity loss）在绝对空间计算，需要对齐:

**修改方案（两版统一）**:
- 在 `m2m_loss.py` 中，计算 position loss 前先减去 root position：
  ```python
  # Before:
  pos_loss = smooth_l1(pred_x1[..., 135:198], target_x1[..., 135:198])
  
  # After (两版统一):
  pred_pos_rel = pred_x1[..., 135:198] - expand_to_joints(pred_x1[..., 0:3])
  target_pos_rel = target_x1[..., 135:198] - expand_to_joints(target_x1[..., 0:3])
  pos_loss = smooth_l1(pred_pos_rel, target_pos_rel)
  ```
- **版本 B** 的 position channels 已经是 relative-to-smooth-root，此公式同样适用（pred_x1[0:3] 就是 smooth_trans）

#### 6.4.2 移除 t² Timestep Weighting

当前 `kimodo_aux_loss.py` 中对辅助 loss 施加了 t² 加权:
```python
if self.timestep_squared_weighting and timesteps is not None:
    t_sq = (timesteps.to(pred_world.device) ** 2)
    per_frame = per_frame * t_sq.unsqueeze(-1)
```

**移除理由**:
1. **KIMODO 不使用 t² 加权**: 经代码审计确认，t² weighting 是我们自行添加的，而非 KIMODO 原版。按"与 KIMODO 对齐"原则移除
2. **Flow matching 特性**: velocity prediction 目标 `v = x1 - x0` 与 t 无关，等权合理
3. **辅助 loss 在低 t 也有信号**: 模型的预测 x1 在任意 t 都有意义，FK 在预测 x1 上计算而非 x_t 上

**修改**: `timestep_squared_weighting=False`（config 修改）

#### 6.4.3 Loss 权重

各 loss 项的权重**待首批实验启动后根据各项 loss 数值量级确定**，暂不硬编码。初始值参考 KIMODO 原版设定，训练后根据 loss 曲线动态调整。

### 6.5 两版配置对比

| 配置项 | 版本 A: SMPL Root | 版本 B: KIMODO Root |
|--------|------------------|-------------------|
| **motion_dim** | 198 | 198 (**相同**) |
| **layout** | `[raw_trans(3), rot6d(132), pos_rel_raw_pelvis(63)]` | `[smooth_trans(3), rot6d(132), pos_rel_smooth_root(63)]` |
| **rotation [3:135]** | 22×6 rot6d | **完全相同** |
| **position loss 空间** | relative to root (对齐后) | relative to smooth root (自然) |
| **t² weighting** | ❌ 移除 | ❌ 移除 |
| **velocity loss** | ✅ | ✅ |
| **FK keypoint loss** | 启用 | 启用 |
| **KIMODO aux losses** | 启用 (权重待定) | 启用 (权重待定) |
| **transl_aug** | ❌ 不启用 (保持纯对比) | ❌ 不需要 (ADMM 已平滑) |
| **数据转换** | 无需转换 | 在线 ADMM 转换 (dataset __getitem__) |
| **额外预处理** | 无 | 仅预算新 mean/std |
| **推理后转换** | 不需要 | smooth_trans 直接作为 SMPL trans (误差 ≤ 6cm) |
| **架构代码改动** | 无 | **无** (维度不变，仅 data transform + mean/std 不同) |

---

## 7. Motion Condition 训练采样分析

### 7.1 概述

Motion condition 采样是 HyMotion M2M 的核心能力之一。它决定了模型在推理时能处理多大范围的 condition pattern。当前有两个采样器版本:

| 采样器 | 使用场景 | 覆盖率 | 机制 |
|--------|---------|--------|------|
| **v2** | caption configs (`cond_mask_prob=0.1`) | ~40% | 两层混合: 60% 参数化 Tier-1 + 40% 模板 Tier-2 |
| **v3** | uncond configs (`cond_mask_prob=0.0`) | ~84% | Rank-K Boolean Tensor Prior, 数学统一 |

### 7.2 V3 Condition Sampler 详解

V3 sampler 是当前最先进的设计，核心是 **Rank-K Boolean Tensor Prior**:

```
M = ⊻_{k=1..K} (t_k ⊗ d_k)

其中:
  K ~ πK = (0.10, 0.55, 0.25, 0.07, 0.03)  对应 K ∈ {0,1,2,3,4}
  t_k ∈ {0,1}^T  — 时间模式 (从 6 种 temporal primitive 中采样)
  d_k ∈ {0,1}^198 — 维度模式 (从 5 种 dimensional kind 中采样)
```

#### 7.2.1 时间分布 (πT: 6 primitives)

| Primitive | 权重 | 覆盖的评估场景 | 分布特征 |
|-----------|------|---------------|---------|
| **all** | 2.0 | E7(first-frame), E8(全帧joint) | 所有帧都有条件 |
| **empty** | 0.3 | T2M (无motion条件) | 所有帧都是生成 |
| **interval** | 3.5 | E2(inbetween), E15(prepend) | 连续时间窗 (prefix/suffix/middle) |
| **periodic** | 4.0 | E3(keyframe@15/30/60), E4(sparse) | 等间隔采样 (5/10/15/20/30/60) |
| **renewal** | 1.5 | 不规则稀疏条件 | Geometric gap i.i.d. |
| **markov** | 1.0 | 连续-间断混合 | 两态 Markov chain |

**时间覆盖分析**:
- ✅ 稠密条件 (>80% 帧): `all` primitive
- ✅ 稀疏条件 (<10% 帧): `periodic` (大步长) + `renewal` (低 ρ)
- ✅ 前缀/后缀: `interval` (1/3 prefix + 1/3 suffix)
- ✅ 等间隔: `periodic` (anchor steps 5/10/15/20/30/60)
- ✅ 随机稀疏: `renewal` + `markov`
- ⚠️ **Gap**: 没有显式的 "multi-segment" primitive（如 [帧0-10] + [帧50-60]），但 K≥2 时两个 `interval` atom 的 OR 可以近似

#### 7.2.2 空间/维度分布 (πD: 5 kinds)

| Kind | 权重 | 覆盖的评估场景 | 锁定的维度 |
|------|------|---------------|-----------|
| **rot_only** | 0.22 | E8(joint rotation control) | 选定关节的 rot6d |
| **pos_only** | 0.30 | E4(end-effector), E6(foot) | 选定关节的 XYZ position |
| **trans_only** | 0.10 | E5(trajectory), translation control | 仅 translation [0:3] |
| **mixed** | 0.18 | 复合条件 | rot + pos + trans 的 OR |
| **all_dim** | 0.20 | E2/E3/E7/E15(全帧锁定) | 全部 198 dims |

**空间覆盖分析**:
- ✅ **全帧全维度**: `all_dim` (20%)
- ✅ **纯旋转条件**: `rot_only` — 解剖学分组 (17 groups, 加权) + Bernoulli + single joint
- ✅ **纯位置条件**: `pos_only` — XYZ 子集采样 (xyz/xz/y/...)
- ✅ **纯轨迹条件**: `trans_only` — translation XYZ 子集
- ✅ **混合模态**: `mixed` — 组合 rot + pos + trans
- ℹ️ **设计说明**: `pos_only` 覆盖 joints 1-21 的 position（维度 [135:198]），不含 pelvis；`trans_only` 覆盖 pelvis translation（维度 [0:3]）。这是**有意的层级分离**——pelvis 由 `trans_only` 专门控制，非 root 关节由 `pos_only` 控制，二者互补覆盖全部空间位置。如需同时控制二者，可用 `mixed` kind 或 K≥2 的多 atom 组合

#### 7.2.3 稀疏度控制

稀疏度由 **K (atom 数量)** 和 **temporal primitive 参数** 共同控制:

| K 值 | 概率 | 典型稀疏度 | 对应任务 |
|------|------|-----------|---------|
| K=0 | 10% | 100% 生成 (无条件) | T2M, unconditional |
| K=1 | 55% | 依 primitive: 1-100% | 大多数 M2M 任务 |
| K=2 | 25% | 多层条件叠加 | 复合约束 (如 trajectory + keyframe) |
| K=3 | 7% | 高密度条件 | 精细编辑/修复 |
| K=4 | 3% | 极高密度 | 接近完整约束 |

**关键**: K≥2 时，多个 atom 的 Boolean OR 产生更丰富的条件模式，这是 v3 优于 v2 的核心——v2 只能产生 v3 K=1 等价的模式子集。

#### 7.2.4 旋转 vs Position 模态覆盖

| 模态 | 支持方式 | 训练采样概率 |
|------|---------|-------------|
| **纯旋转条件** | `rot_only` kind (22%) | ~22% × (1-0.10) ≈ 20% |
| **纯位置条件** | `pos_only` kind (30%) | ~30% × (1-0.10) ≈ 27% |
| **纯轨迹条件** | `trans_only` kind (10%) | ~10% × (1-0.10) ≈ 9% |
| **混合模态** | `mixed` kind (18%) + K≥2 的 cross-kind | ~18% + 多 atom 组合 |
| **全模态** | `all_dim` kind (20%) | ~20% × (1-0.10) ≈ 18% |

### 7.3 V2 vs V3 对比与 Caption 配置的 Gap

| 维度 | V2 (caption configs) | V3 (uncond configs) |
|------|---------------------|-------------------|
| **时间分布** | 固定模板 (inbetween, keyframe, etc.) | 6 primitives, 连续参数 |
| **空间分布** | 预定义 joint groups | 17 anatomical groups + Bernoulli + single |
| **稀疏度** | 离散固定 (由模板决定) | K 控制, 连续可调 |
| **模态** | 全维度锁定为主 | rot/pos/trans 独立可控 |
| **覆盖率** | ~40% | ~84% |
| **扩展性** | 加新模板 | 加新 primitive/kind |

**Caption 配置使用 V2 的原因**: 历史原因，v2 sampler 在 caption configs 出现时是最新版本。
**建议**: 将 caption configs 也升级到 v3 sampler，统一训练采样。

### 7.4 当前方案的不足与改进方向

| 不足 | 严重度 | 改进方向 |
|------|--------|---------|
| Caption configs 仍用 v2 sampler (40% 覆盖) | P1 | 统一使用 v3 |
| 无显式 multi-segment temporal primitive | P2 | K≥2 的 interval OR 近似覆盖 |
| ~~Pelvis position 不在 `pos_only` 覆盖中~~ | ~~P2~~ | 已有 `trans_only` 覆盖 translation（层级互补设计，非 gap） |
| 无 contact/foot height 条件类型 | P2 | 后续实验: 增加 foot contact channel |
| 无 velocity 条件类型 | P3 | 需要扩展 dimensional kind |

### 7.5 V3 Sampler 已知缺陷的坦诚评估 [v1.7 新增]

> 对应反馈：「你觉得V3的采样器已经没有缺陷了吗？」

**V3 Sampler 当前存在以下已知缺陷/风险：**

1. **条件密度分布偏移 (P1)**: V3 的 K 值分布 (K=0:10%, K=1:55%, K=2:25%, K=3:7%, K=4:3%) 是手动设定的先验，未经训练验证其最优性。实际应用中，高密度条件（如 dense edit/repair）和低密度条件（如单 keyframe）的使用频率可能与训练分布不匹配。如果 K 分布偏离实际使用分布，模型在 OOD 密度下的表现可能显著退化。

2. **Primitive 参数空间的 coverage gap (P1)**: 6 种 temporal primitive 虽然覆盖率从 V2 的 ~40% 提升到 ~84%，但仍有 ~16% 的条件模式无法产生。例如:
   - **不规则间隔的多段条件** (如帧 [10-20, 50-60, 100-120])：K≥2 的 interval OR 虽可近似，但精确匹配概率低
   - **单帧散列条件** (如仅约束帧 5, 23, 67, 121)：只有 `scatter` primitive 覆盖，但 scatter 的 Bernoulli 采样难以精确控制帧数
   - **渐进密度条件** (如前半段 dense、后半段 sparse)：需要 K≥2 且恰好选到互补 primitive 的组合

3. **Dimensional kind 粒度不足 (P2)**: 5 种 dimensional kind (all_dim, rot_only, pos_only, trans_only, mixed) 对关节的控制粒度是 group 级别（17 个预定义解剖学 group），无法精确表达"仅约束右手腕旋转+左膝位置"这类细粒度跨模态条件。虽然 K≥2 的多 atom OR 理论上可以组合出此类模式，但采样到的概率极低。

4. **训练与推理的 mask 分布一致性未验证 (P1)**: V3 sampler 的条件 mask 分布在训练时由采样器随机生成，但推理时由用户指定确定性 mask。训练采样分布是否充分覆盖了实际推理时会遇到的 mask 模式，目前仅靠 coverage 指标 (~84%) 估计，没有在真实任务上系统验证。具体风险:
   - 推理时的精确 keyframe 约束 (例如帧 0 和帧 196 的 exact keypose) 在训练中的出现频率是否足够？
   - 极端稀疏 (仅 1-2 帧) 和极端密集 (>90% 帧约束) 两个尾部的表现是否稳定？

5. **缺少 adaptive/learned 采样 (P3)**: 当前 V3 是纯规则采样，所有 primitive 权重和 K 分布都是固定的。理想情况下可以根据训练 loss landscape 自适应调整采样分布（类似 curriculum learning），但这增加了实现复杂度，暂不在 Phase 0 考虑。

**结论**: V3 Sampler 相比 V2 是一个显著进步（覆盖率 40%→84%，支持模态独立控制），但**不是一个完善的方案**。上述缺陷 1 和 4 可能直接影响首批实验的效果——如果训练采样分布与评估任务分布差距过大，模型在特定任务上的表现可能不如预期。建议在 E1-E4 评估时重点关注 coverage gap 对应的任务类型（单帧 keyframe、多段条件），并根据评估结果调整 V3 参数。

---

## 8. HyMotion M2M 支持的任务枚举 [v1.6 新增]

HyMotion M2M 是一个**统一的 motion-to-motion 框架**，通过 VACE 条件化 + mask 模式的组合，用单一模型覆盖以下全部任务。任务按类别分组:

### 8.1 生成类任务

| 任务 ID | 任务名 | 描述 | 条件输入 | 对应 Condition Sampler |
|---------|--------|------|---------|---------------------|
| **E1** | Text-to-Motion (T2M) | 纯文本驱动生成，无 motion 条件 | text caption | K=0 (empty mask) |
| **E13** | Multi-Prompt Generation | 给定 N 段文本描述，自回归链式生成任意长度动作 | N × text captions + 上一段末尾帧 | K=1 (interval: prefix anchor) |

### 8.2 Motion Completion 任务（时间维度约束）

| 任务 ID | 任务名 | 描述 | 条件输入 | 对应 Condition Sampler |
|---------|--------|------|---------|---------------------|
| **E2** | Motion In-Betweening | 给定首尾 N 帧，生成中间过渡 | 首/尾各 N 帧 (all_dim) | K=1 (interval: prefix+suffix) |
| **E3** | Keyframe Interpolation | 给定稀疏关键帧（等间隔或自适应），插值生成完整动作 | 每 K 帧锁定 (all_dim) | K=1 (periodic) |
| **E7** | First-Frame Continuation | 给定第一帧 + 文本，续写后续动作 | frame 0 (all_dim) + text | K=1 (interval: prefix) |
| **E14** | Transition Stitching | 拼接两段动作 A→B，生成自然过渡 | A 末尾帧 + B 起始帧 (all_dim) | K≥2 (两个 interval) |
| **E15** | Prepend to Start Pose | 给定完整动作 A 和目标起始姿态 P，在 A 前生成过渡帧 | P (frame 0) + A (suffix) | K=1 (interval) |
| **E8** | Loop Animation | 生成循环动作，首尾帧一致 | frame 0 = frame T (all_dim) | K=1 (interval) |

### 8.3 Motion Editing 任务（空间维度约束）

| 任务 ID | 任务名 | 描述 | 条件输入 | 对应 Condition Sampler |
|---------|--------|------|---------|---------------------|
| **E4** | End-Effector Constraint | 锁定末端效应器（手/脚）的世界坐标位置，生成满足约束的动作 | 稀疏帧的 joint position (pos_only) | K=1 (periodic + pos_only) |
| **E5** | Trajectory Following | 跟随 root XZ 轨迹生成动作 | pelvis XZ at condition frames (trans_only) | K=1 (periodic + trans_only) |
| **E6** | Foot Ground Constraint | 在脚-地面接触帧锁定脚踝位置 | ankle position at contact frames (pos_only) | K=1 (renewal/periodic + pos_only) |
| **E10** | Part-Level Control | 锁定指定身体部位的旋转，重新生成其余部分 | 指定关节的 rot6d (rot_only) | K=1 (all + rot_only subset) |

### 8.4 Motion Repair 任务

| 任务 ID | 任务名 | 描述 | 条件输入 | 对应 Condition Sampler |
|---------|--------|------|---------|---------------------|
| **E9** | Motion Repair | 修复质量检测器标记的缺陷帧（抖动/滑步/穿模） | 非缺陷帧 (adaptive mask from checker) | K≥1 (checker-driven adaptive mask) |

### 8.5 任务覆盖与 Condition Sampler 映射

上述 14 个任务可归纳为以下训练时条件模式的组合:

```
任务覆盖 = {时间模式} × {维度模式}

时间模式 (§7.2.1):
  all      → E8(loop), E10(part-level)
  empty    → E1(T2M)
  interval → E2(inbetween), E7(first-frame), E13(multi-prompt), E14(transition), E15(prepend)
  periodic → E3(keyframe), E4(end-effector), E5(trajectory), E6(foot)
  renewal  → E6(foot, 不规则接触)
  markov   → E9(repair, checker-driven segments)

维度模式 (§7.2.2):
  all_dim   → E1, E2, E3, E7, E8, E13, E14, E15
  rot_only  → E10(part-level)
  pos_only  → E4(end-effector), E6(foot)
  trans_only → E5(trajectory)
  mixed     → 复合条件 (e.g., trajectory + keyframe)
```

**关键结论**: V3 Condition Sampler 的 6 种时间 primitive × 5 种维度 kind 的组合已经覆盖了全部 14 个评估任务。K≥2 的多 atom 叠加进一步支持任意复合条件。

---

## 9. 实验计划（按优先级排序） [v1.5 重写, v1.6 修正维度/null_embedding]

### 9.1 实验设计原则

每个实验只改变一个模块，按照 **改进生效概率** 从高到低排序。这样即使后续实验失败，前面的成功实验仍然可用。

### 9.2 首批实验（4 个 Taiji 任务）

首批实验目标是验证 **Root 表征** 和 **Loss 对齐** 的效果。所有实验共享以下 loss 配置:

```python
# 统一 loss 配置 (两版共用)
velocity_weight = 1.0
motion_smoothness_weight = 0.5
trans_dim_weight = 5.0
keypoints3d_weight = 10.0          # D1 修复: 启用 FK loss
timestep_squared_weighting = False  # 移除 t² weighting
aux_joint_pos_weight = 50.0
aux_joint_vel_weight = 500.0
aux_fk_consistency_weight = 1500.0
# position loss 统一在 relative-to-root 空间计算
# transl_aug_prob: 两版均不启用 (§1.2.3: 保持纯 A/B 对比)
```

#### 实验 E1: SMPL Root + Uncond (Baseline)

| 项目 | 值 |
|------|-----|
| **Root 表征** | 版本 A: SMPL Root (198-dim) |
| **训练任务** | Unconditional (cond_mask_prob=0.0) |
| **Condition Sampler** | v3 |
| **修改内容** | 启用 FK loss + 移除 t² + position loss relative-to-root |
| **Config 基础** | `hymotion_m2m_v2_uncond_local_046b.py` |
| **GPU** | 48 × V100 (6 nodes × 8 GPU) |
| **预期效果** | 滑步改善 (FK loss)，baseline quality 提升 |
| **有效概率** | 90% (FK loss 是 KIMODO 验证过的已知有效方法) |

#### 实验 E2: SMPL Root + Caption

| 项目 | 值 |
|------|-----|
| **Root 表征** | 版本 A: SMPL Root (198-dim) |
| **训练任务** | Caption (cond_mask_prob=0.1) |
| **Condition Sampler** | v3 (从 v2 升级) |
| **修改内容** | 同 E1 + v2→v3 sampler 升级 + `null_embedding_source` 配置确保 CFG 正确 |
| **Config 基础** | `hymotion_m2m_v2_caption_local_046b.py` |
| **GPU** | 48 × V100 |
| **预期效果** | Text conditioning 恢复 + 滑步改善 |
| **有效概率** | 85% |

#### 实验 E3: KIMODO Root + Uncond

| 项目 | 值 |
|------|-----|
| **Root 表征** | 版本 B: KIMODO Root (198-dim) |
| **训练任务** | Unconditional (cond_mask_prob=0.0) |
| **Condition Sampler** | v3 (适配 198-dim) |
| **修改内容** | 新 root 表征 (ADMM smooth translation 替换 raw translation [0:3]) + 在线转换 + 新 mean/std |
| **Config 基础** | 新建 `hymotion_m2m_v2_kimodo_uncond_046b.py` |
| **GPU** | 48 × V100 |
| **预期效果** | 在 E1 基础上进一步减少滑步（ADMM smooth translation 效果） |
| **有效概率** | 70% (KIMODO 论文验证有效，但迁移到我们的 SMPL-22 表征可能有 gap) |

#### 实验 E4: KIMODO Root + Caption

| 项目 | 值 |
|------|-----|
| **Root 表征** | 版本 B: KIMODO Root (198-dim) |
| **训练任务** | Caption (cond_mask_prob=0.1) |
| **Condition Sampler** | v3 (适配 198-dim) |
| **修改内容** | 同 E3 + caption 支持 + `null_embedding_source` 配置 |
| **Config 基础** | 新建 `hymotion_m2m_v2_kimodo_caption_046b.py` |
| **GPU** | 48 × V100 |
| **预期效果** | KIMODO root 下的 text 条件生成 |
| **有效概率** | 65% |

### 9.3 资源需求

| 资源 | 需求 |
|------|------|
| **GPU 总量** | 4 × 48 = 192 卡 V100 |
| **来源** | DHC_DD 或 DHA 应用组 (哪里空用哪里) |
| **停止的任务** | uncond_local, caption_local (当前跑的两个实验) |
| **预计训练时长** | 每个实验 ~5-7 天 (100K steps @ batch_size=20-28) |

### 9.4 后续实验（根据 E1-E4 结果决定）

| 实验 | 条件 | 内容 | 有效概率 |
|------|------|------|---------|
| **E5: + PDCT Curriculum** | E2/E4 caption 完成 | 从 E2/E4 checkpoint 继续，用 PDCT 低密度→高密度课程训练 (text pathway 强化) | 65% |
| **E5b: + STP 语义时空规划** | E2/E4 caption 完成 | 从 E2/E4 checkpoint 继续，启用 GTA 接地 caption (gta_prob=0.5) + CAPT 规划头辅助损失 | 70% |
| **E5c: + CPOS Inference** | E2/E4 caption 完成 | 不重训，直接在推理端使用 CPOS schedule (text-heavy → motion-heavy CFG) | 55% |
| **E5d: PDCT + STP + CPOS** | E5/E5b 完成 | 三者联合: PDCT 课程 + STP 接地规划 + CPOS 推理，验证互补效果 | 75% |
| **E6: + Explicit Velocity Channel** | E3 > E1 (KIMODO root 有效) | 在版本 B 基础上增加 21×3 joint velocity 通道 | 60% |
| **E7: + Foot Contact Channel** | E1 滑步仍未解决 | 增加 4-dim foot contact 信号 | 55% |
| **E8: + FK Keypoint Loss** | E1/E3 完成 | 启用 keypoints3d_weight 消融滑步改善 | 70% |

### 9.5 评估标准

实验成功的**最低标准**:
1. ✅ 在 Taiji 上成功启动
2. ✅ Loss 正常下降（前 1000 steps loss 单调下降趋势）
3. ✅ 无 NaN/Inf
4. ✅ Gradient norm 在合理范围内 (< 100)

实验效果的**对比标准** (训练 ~50K steps 后):
1. Foot Skating Score 对比 (E1 vs E3, E2 vs E4)
2. FID / R-Precision 对比
3. MPJPE 对比
4. 主观可视化对比

---

## 10. 评估指标与任务覆盖 [v1.5 从原 §7.2/§7.3 迁移]

### 10.1 评估指标

#### Text Conditioning 评估
| 指标 | 说明 | 目标 |
|------|------|------|
| **R-Precision (Top-1/2/3)** | Text-motion 匹配准确率 | > 0.45 / 0.65 / 0.75 |
| **MM-Dist** | Text-motion 特征距离 | < 3.5 |
| **Text Effect Ratio** | ‖pred_w_text - pred_wo_text‖ / ‖pred_wo_text‖ | > 0.15 |
| **FID** | 生成动作的分布质量 | < 0.5 |

#### Motion Condition 评估
| 指标 | 说明 | 目标 |
|------|------|------|
| **Condition MPJPE** | 条件帧处的关节位置误差 | < 1mm |
| **Boundary Smoothness** | 条件/生成边界的加速度跳变 | < 50 mm/s² |
| **Interpolation Quality** | 中间帧的插值准确度 | MPJPE < 20mm |

#### 质量评估
| 指标 | 说明 | 目标 |
|------|------|------|
| **Foot Skating** | 接触帧脚部滑动距离 | < 0.25 |
| **Jitter** | 关节位置高频抖动 | < 600 |
| **Physical Plausibility** | 脚穿地面、悬空等比例 | < 5% |

### 10.2 评估任务覆盖

全部 E1-E15 任务 + 新增:
- **E-Text**: 纯 T2M 质量评估
- **E-TextCond**: Text + sparse condition 联合评估
- **E-Edit**: 语义编辑（"让走路变得更高兴"）评估
- **E-Skating**: 滑步专项评估 (foot skating score, FK keypoint error)

---

## 11. 与前沿方法对比及新颖性分析

### 11.1 方法对比矩阵

| 特性 | 当前 M2M | VACE (Wan2.1) | OmniGen2 | Seedance 2.0 | Step1X-Edit | **CDO-FM (Ours)** |
|------|---------|---------------|----------|--------------|-------------|-------------------|
| Text-Motion 平衡 | ✗ (竞争) | ✗ | ✓ (双路径) | ✓ (DB-DiT) | ✓ (MLLM路由) | **✓ (PDCT 课程训练 + STP 语义时空规划)** |
| 零额外参数 | ✓ | ✓ | ✗ (双路径) | ✗ (DB-DiT) | ✗ (MLLM) | **✓** |
| 推理时条件解耦 | ✗ | ✗ | ✗ | ? | ✗ | **✓ (CPOS 渐进 ODE)** |
| Root 表征鲁棒性 | ✗ (raw MoCap) | N/A | N/A | N/A | N/A | **✓ (Dual Root: ADMM平滑)** |
| 多任务条件采样 | ✗ (简单随机) | ✗ | ✗ | ✗ | ✗ | **✓ (V3 Rank-K 张量先验)** |
| 任务统一 | 部分 | ✓ | ✓ | ✓ | 编辑为主 | **✓ (全覆盖)** |
| 运动领域适配 | ✓ | ✗ (视频) | ✗ (图像) | ✗ (视频) | ✗ (图像) | **✓** |

### 11.2 新颖性论证

1. **Progressive Density Curriculum Training (PDCT)**: 不同于 OmniGen2 的双路径架构或 Seedance 的 DB-DiT（均引入额外参数），PDCT 仅通过训练时的条件密度分布调度就实现了 text/motion 平衡——核心洞察是利用信息密度的时序不对称性：先在低密度条件下迫使 text pathway 建立（类似 information bottleneck），再逐步引入高密度条件。这是首次在 motion generation 中将 curriculum learning 与条件密度分布显式关联。

2. **Dual Root Representation**: 提出 SMPL Root 与 KIMODO Root (ADMM-smoothed trajectory) 双版本方案，首次系统化地研究 root representation 对 motion generation 滑步问题的影响。两种表征共享完全相同的 198-dim 框架和 loss 设计，仅 translation [0:3] 和 position 参考系不同，可控变量地验证 smooth trajectory 对生成质量的提升。

3. **Rank-K Boolean Tensor Prior (V3 Condition Sampler)**: 提出基于 6 种时间原语 × 5 种维度类型的结构化条件采样策略，相比 random Bernoulli mask 能更系统地覆盖 motion generation 的多样化任务空间（keyframe, prefix, suffix, inbetween, outpainting 及其组合），是首次在统一 M2M 框架中设计面向任务覆盖的条件采样分布。

4. **Semantic-Temporal Planning (STP)**: 揭示了 shortcut learning 的根本原因不在于 dropout 分配或数据构造，而在于模型缺乏**对文本语义到时空区域映射的显式推理能力**。STP 通过三组件协同——接地文本增强 (GTA) 让模型学习文本-时空对齐、条件感知规划标记 (CAPT) 迫使 backbone 编码文本-条件-时空三角关系、规划引导生成 (PGG) 在推理时按规划图自适应调度引导强度——使模型先"想清楚"文本各语义片段对应的时空区域和条件覆盖关系，再执行生成。这一机制**与具体任务类型无关** (T2M/M2M/inpainting/editing 统一适用)。

5. **Condition-Progressive ODE Sampling (CPOS)**: 发现 flow matching ODE 的粗-细粒度生成特性与 text（全局语义）/ motion condition（局部结构）的信息粒度天然对应，首次提出将条件引导强度与 ODE timestep 解耦调度——在 ODE 早期放大 text CFG、弱化 motion condition 约束，后期反转。这是一种 training-free 的推理改进，可即时应用于任何已训练的 conditional flow matching 模型。

### 11.3 与 VACE 的关系

CDO-FM 可以视为 VACE 框架在 motion generation 领域的**深度进化**:
- VACE 提出了 `V=[T;F;M]` 的统一编码 → v2 进化为 MAN + `no_inactive` 的 3 通道编码 `[x_t(MAN), reactive, mask]`，条件信息直接编码进 x_t，并通过 V3 Condition Sampler 实现更结构化的条件分布采样
- VACE 使用固定的 text dropout 训练 → 我们的 STP 通过接地文本增强和规划头辅助训练让模型显式学习文本-时空-条件的三角关系，PDCT 在训练时间维度编排条件密度分布
- VACE 的 single-scale CFG → 我们的 CPOS 实现了 ODE timestep 自适应的条件引导 schedule
- **核心差异**: CDO-FM 的所有创新（PDCT, STP, CPOS）均为低额外参数的训练/推理策略 (STP 的 planning head 仅 <0.01% 参数)，核心模型架构不变，可直接应用于任何基于 VACE 框架的模型

---

## 12. 顶会论文定位

### 12.1 推荐标题

**"CDO-FM: Condition-Decoupled Orchestration for Unified Text-and-Motion Conditioned Human Motion Generation"**

或更简洁:

**"MotionCanvas: Density-Aware Condition Orchestration for Universal Motion Generation"**

### 12.2 故事线

> 现有 motion generation 方法要么只做 text-to-motion (T2M)，要么只做 motion completion (M2M)，无法在一个模型中同时理解文本语义和空间运动约束。我们发现根因在于两类条件的**信息密度天然不对称**——一句话的信息量远低于 10 帧 dense keyframe。简单地将两者混入同一 conditioning pipeline 会导致模型学到忽略文本的 shortcut。
>
> 为此，我们提出 CDO-FM，通过 (1) 渐进密度课程训练 (PDCT) 在训练阶段编排条件密度分布，先在低密度条件下迫使 text pathway 建立再逐步引入高密度条件，(2) 语义时空规划 (STP) 让模型在生成前先推理文本各语义片段对应的时空区域和条件覆盖关系——通过接地文本增强 (GTA) 学习文本-时空对齐、条件感知规划标记 (CAPT) 编码三角关系、规划引导生成 (PGG) 自适应调度引导强度，(3) Dual Root 表征方案 (SMPL Root + KIMODO ADMM-smoothed Root) 系统性解决 MoCap 噪声导致的滑步问题，(4) 结构化条件采样策略 (Rank-K Boolean Tensor Prior) 系统覆盖多样化 M2M 任务空间，(5) 条件渐进 ODE 采样 (CPOS) 在推理时根据 ODE timestep 自适应调度 text/motion 引导强度。STP 的 planning head 仅增加 <0.01% 参数，其余创新均为零额外参数的训练/推理策略。
>
> 在 XXX benchmark 上，CDO-FM 首次在单一模型中同时达到 T2M SOTA 和 M2M SOTA，且在 text-conditioned motion completion 这一新任务上显著优于所有 baseline。

### 12.3 目标会议

- **首选**: CVPR 2027 (DDL ~Nov 2026) / ICLR 2027 (DDL ~Oct 2026)
- **备选**: NeurIPS 2027 (DDL ~May 2027) / ECCV 2027 (DDL ~Mar 2027)

### 12.4 可能的审稿人关注点

| 审稿人质疑 | 预备回应 |
|-----------|---------|
| PDCT curriculum 阶段划分是否 ad hoc？ | 消融实验: 直接全分布训练 vs PDCT 三阶段，并提供 information bottleneck 理论支撑 |
| STP 的 GTA 接地标注质量是否可靠？CAPT planning head 是否真正有效？ | GTA 标注由 LLM 生成，提供标注质量评估 (人工抽样 200 条)；CAPT 消融: 有/无 planning head 的 text effect ratio 和 R-Precision 对比 |
| Dual Root 的 ADMM 平滑是否只是简单预处理？ | 提供消融: raw trans vs smooth trans 的 FID/foot skating 对比 |
| V3 Condition Sampler 相比 random mask 提升多大？ | V2 vs V3 sampler 消融 + 任务覆盖分析 |
| CPOS 的 bell-shaped/sigmoid schedule 是否 heuristic？ | 提供与 constant CFG、linear schedule 的对比消融；引用 ODE coarse-to-fine 理论支撑 |
| STP planning head 引入了额外参数，是否破坏了通用性？ | Planning head 仅一个线性层 (<0.01% 总参数)，可以 training-free 移除 (PGG 退化为 CPOS)；实验证明 PGG 在移除 CAPT 后仍优于 baseline |
| 198-dim motion representation 是否限制了方法通用性？ | 讨论扩展到 SMPL-X/手部的路径 |

---

## 13. 风险与备选方案

### 13.1 风险评估

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|---------|
| PDCT 课程训练阶段切换时 loss 震荡 | 中 | 中 | Phase B 使用线性 ramp 而非阶梯切换，Phase 过渡设置 warmup |
| STP-GTA 接地标注质量不均导致模型学习到错误的时空对齐 | 低-中 | 中 | gta_prob 控制接地 caption 使用比例；GTA 标注可后续人工校验；CAPT 有独立 GT 不依赖 GTA 质量 |
| KIMODO Root (ADMM平滑) 对生成质量提升有限 | 低-中 | 中 | 退化为 SMPL Root (版本 A)，E1/E2 消融直接验证 |
| V3 Condition Sampler 对 caption 模型效果不明显 | 低 | 低 | 回退到 V2 sampler (random Bernoulli mask) |
| CPOS schedule 参数 (σ_text, k_motion 等) 需要调优 | 中 | 低 | 基于 ODE 理论提供合理默认值，grid search 仅 2-3 个超参 |

### 13.2 降级方案

如果完整 CDO-FM 的某些组件效果不理想，可以独立使用其中任一子系统：

**最小可行方案 (MVP)**: Dual Root (SMPL + KIMODO) + V3 Sampler + CPOS
- 预期: 解决滑步问题 + 多任务条件覆盖 + 推理时 text/motion 平衡
- 成本: 最低，CPOS 为 training-free 推理策略，无需重新训练
- 时间: 1 周 (CPOS 仅需实现推理 schedule)

**中间方案**: MVP + STP-GTA
- 预期: 接地文本增强让模型学习文本-时空对齐，改善 text 理解能力
- 成本: LLM 标注现有 caption (一次性预处理)，从 E1-E4 checkpoint 继续训练
- 时间: 2-3 周 (含 GTA 标注生成)

**完整方案**: MVP + STP (GTA+CAPT+PGG) + PDCT
- 预期: 最佳性能——PDCT 先建立 text pathway，STP 全链路引入语义-时空推理 + CPOS/PGG 推理优化
- 成本: 需要额外 20K-30K 步训练 (Phase B + Phase C with STP)
- 时间: 4-6 周

---

## 14. 实施路线图 [v1.7 更新]

### Phase 0: Dual Root + Loss 对齐 + 首批实验 (Week 1-2) — v1.5 核心

```
✅ v2 caption bugs 修复 (2026-03-27 ~ 2026-05-12):
   - B1/B2-ext: bundle params 保存 + null embedding 加载链修复 (影响 v2 caption resume/phase/soar)
   - B5/B6: text token OOD + null embedding 分布 (影响 v2 caption)
   - 6 个 v2 caption 配置已添加 null_embedding_source (phase1/phase2/soar × local/global)
✅ v2 uncond_local: 无 bug 影响，无需修复
□ 实现 KIMODO Root 表征 (§6):
  □ 实现在线转换函数 smpl_trans_to_smooth_trans() 和 smooth_trans_to_smpl_trans()
     (仅替换 translation [0:3] 为 ADMM平滑, rotation 和 position 透传)
  □ 实现或集成 ADMM XZ 平滑 (margn ≤ 6cm, 可从 KIMODO 或自实现)
  □   □ 数据预处理: SMPL root 在线转换为 KIMODO root (198-dim) + 计算新 mean/std
  □ 单元测试: 在线转换验证 (SMPL 198 → 在线ADMM平滑 → SMPL 198 roundtrip, 零误差 + 6cm tolerance)
□ Loss 对齐 (§6.4):
  □ Position loss 改为 relative-to-root 空间
  □ 移除 t² timestep weighting (config 修改)
□ 滑步修复 — config-only (§1.2.3):
  □ 启用 FK keypoint loss: keypoints3d_weight=10.0 (D1)
□ Config 准备:
  □ E1: hymotion_m2m_v2_smpl_uncond_046b.py (版本 A, 198-dim)
  □ E2: hymotion_m2m_v2_smpl_caption_046b.py (版本 A + caption + null_embedding_source)
  □ E3: hymotion_m2m_v2_kimodo_uncond_046b.py (版本 B, 198-dim, ADMM平滑translation)
  □ E4: hymotion_m2m_v2_kimodo_caption_046b.py (版本 B + caption + null_embedding_source)
□ Debug on lzy_debug_machine_1/2:
  □ 单步训练 (版本 A 和版本 B 各 1 step)
  □ 推理测试 (版本 B 输出 → SMPL 转换 → 可视化)
□ 停止当前 uncond_local、caption_local 实验
□ 在 Taiji 提交 E1-E4 (每个 48×V100)
□ 确认 loss 正常下降
```

### Phase 1: 语义时空规划 — STP + PDCT + CPOS (Week 3-5)

```
□ 根据 E1-E4 结果确定 SMPL Root vs KIMODO Root 的最优方案
□ STP-GTA: 利用 LLM 对训练集 caption 生成时空接地标注 (grounded_caption)
□ STP-CAPT: 实现 PlanningHead 轻量规划头 + 辅助训练损失
□ 从 E1-E4 checkpoint 出发，以 STP-GTA+CAPT 继续训练 (E5b)
□ 实现 PDCT 三阶段 schedule: Phase A/B/C 的 V3 K-distribution 切换
□ 从 E1-E4 checkpoint 出发，以 PDCT curriculum 继续训练 (E5)
□ 实现 CPOS 推理 schedule: bell-shaped text CFG + sigmoid replacement alpha
□ 在 E1-E4 基础 checkpoint 上直接测试 CPOS (E5c, training-free)
□ STP-PGG: 实现规划引导生成——将 CAPT 规划图反馈到 CPOS schedule
□ 切换训练数据到 high_quality.json (456K)
□ 训练完整组合: PDCT + STP + CPOS (E5d)
□ 评估: 确认 text conditioning 质量提升 + 规划图可视化验证，对比各消融
```

### Phase 2: 质量增强 — 滑步专项 + 数据质量 (Week 5-7)

```
□ 根据 E1-E4 KIMODO Root 结果评估滑步改善
□ 如 KIMODO Root 有效，后续实验统一切换
□ 启用 FK keypoint loss 消融
□ 探索 foot contact channel (后续实验)
□ 训练最终版本
```

### Phase 3: 论文撰写 (Week 7-10)

```
□ 全面评估: E1-E15 + 新增评估任务
□ 可视化: 生成 demo videos, attention maps
□ 论文撰写: Introduction, Method, Experiments, Ablation
□ 用户研究 (可选)
□ 提交
```

---

## 附录 A: 关键代码修改清单

### Phase 0 Bug Fixes（已完成 2026-05-12）

**v2 caption 相关修复（B1/B2-ext/B5/B6）：**

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `hftrainer/runner/accelerate_runner.py` | 修改 | 新增 `_patch_zero_null_embeddings_from_pretrained()`, 在 auto_resume 和 load_from 后调用 (B2-ext) |
| `configs/hymotion_m2m_v2/` × 6 configs | 修改 | phase1/phase2/soar caption configs 添加 `null_embedding_source` (B2-ext) |

> **注**：B3 (VACE reactive 泄漏) 仅影响 v1 (`split_reactive` 模式)；v2 从设计之初即使用 `no_inactive` 模式，不存在该问题。B4 (`cond_mask_prob` 默认值) v2 config 显式覆盖，无实际影响。两者均不列入 v2 修复清单。

### CDO-FM 零参数训练策略（待实施）

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `hftrainer/models/motion/hymotion_m2m/bundle.py` | 修改 | 增加 KIMODO Root 在线转换支持 |
| `hftrainer/trainers/motion/hymotion_m2m_trainer.py` | 修改 | Position loss 改为 relative-to-root; PDCT schedule 支持 |
| `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` | 修改 | CPOS 渐进 ODE 采样 (bell-shaped text CFG + sigmoid replacement alpha) |
| `hftrainer/datasets/motion/transforms/` | 修改 | PrepareM2MCondition 集成 ADMM 在线转换 |
| `hftrainer/datasets/motion/` | 修改 | 支持 GTA 接地 caption 加载 (grounded_caption_path); 以 gta_prob 概率切换标准/接地 caption |
| `hftrainer/models/motion/hymotion_m2m/bundle.py` | 修改 | 新增 PlanningHead 轻量规划头; planning loss 计算 |
| `configs/hymotion_m2m_v2/` | 新增/修改 | PDCT phase configs (Phase A/B/C 的 V3 K-distribution schedule), STP-GTA/CAPT 配置 |

## 附录 B: 参考文献

1. **VACE** (Alibaba, 2025): All-in-One Video Creation and Editing — VCU 统一编码, Concept Decoupling
2. **OmniGen2** (BAAI, 2025): Decoupled dual-path, Omni-RoPE, reflection data
3. **Seedance 2.0** (ByteDance, 2026): DB-DiT, omnireference, physics-aware rendering
4. **Step1X-Edit** (StepFun, 2025): MLLM→DiT pipeline, noise-augmented training
5. **UniCombine** (Fudan/Tencent, 2025): Conditional MMDiT Attention, LoRA switching
6. **Flux Fill** (BFL, 2024): Mask concatenation inpainting, guided distillation
7. **MotionCLR** (2024): Attention-based motion diffusion, training-free editing
8. **MotionGPT-2** (NeurIPS 2024): VQ-tokenized motion + LLM

## 附录 C: 计算资源预算

### C.1 首批实验 (Phase 0, v1.5)

| 实验 | GPU 配置 | 预计时长 | 总 GPU 小时 |
|------|---------|---------|------------|
| E1: SMPL Root + Uncond | 48×V100 (6 nodes) | 120-168h | 5,760-8,064 |
| E2: SMPL Root + Caption | 48×V100 (6 nodes) | 120-168h | 5,760-8,064 |
| E3: KIMODO Root + Uncond | 48×V100 (6 nodes) | 120-168h | 5,760-8,064 |
| E4: KIMODO Root + Caption | 48×V100 (6 nodes) | 120-168h | 5,760-8,064 |
| **首批合计** | **192 卡** | | **~23,040-32,256 GPU·h** |

### C.2 后续实验 (Phase 1-3)

| 实验 | GPU 配置 | 预计时长 | 总 GPU 小时 |
|------|---------|---------|------------|
| E5-E10: 消融实验 | 8-32×V100 | 72-168h each | ~10,000 |
| 评估 + 其他 | 8×V100 | 48h | 384 |
| **后续合计** | | | **~10,384 GPU·h** |

---

## 附录 D: KIMODO Root 转换规范 [v1.7 重写]

### D.1 维度映射

```
SMPL Root (198-dim) → KIMODO Root (198-dim) 在线转换:

SMPL Root (版本 A):                  KIMODO Root (版本 B):
[0:3]   raw_trans      ──ADMM──→   [0:3]    smooth_trans = ADMM(raw_trans.xz) + raw_trans.y
[3:135] rot6d (22×6)   ──透传──→   [3:135]  rot6d (22×6)  (不变)
[135:198] pos_rel_raw_pelvis ──重算──→ [135:198] pos_rel_smooth_root
          = FK(rot)[j] - raw_pelvis          = FK(rot)[j] - smooth_trans

逆转换 (KIMODO 198 → SMPL 198):
需要原始 raw_trans (推理时不可用，因此版本 B 推理直接输出 smooth_trans，
再用 FK 重建 raw pelvis position)。
实际策略: 训练时 ADMM 在线转换; 推理时版本 B 输出即最终结果。
```

**关键变化**: 相比 v1.6，v1.7 **不再使用** trans_residual，维度保持 198 不变。
两个版本的唯一区别是 [0:3] translation channel 和 [135:198] position 的参考系原点。

### D.2 ADMM 平滑参数

| 参数 | 值 | 说明 |
|------|-----|------|
| margin | 0.06m (6cm) | 平滑 XZ 轨迹的最大偏移约束 |
| step_size | 0.25 × sqrt(diag_max) | 自适应步长 |
| iterations | 100 per level | 每层优化迭代数 |
| smoothed_axes | XZ only | Y 轴高度保持原始值 |

### D.3 转换函数实现

```python
def smpl_trans_to_smooth_trans(raw_trans: Tensor, admm_margin: float = 0.06) -> Tensor:
    """
    在线转换: SMPL raw translation → KIMODO smooth translation.
    仅平滑 XZ 分量, Y 保持不变。
    
    Args:
        raw_trans: (B, T, 3) or (T, 3) — 原始 MoCap translation
        admm_margin: ADMM 平滑的最大偏移约束 (默认 6cm)
    Returns:
        smooth_trans: (B, T, 3) or (T, 3) — 平滑后的 translation
    """
    smooth_trans = raw_trans.clone()
    smooth_trans[..., [0, 2]] = admm_smooth_xz(raw_trans[..., [0, 2]], margin=admm_margin)
    return smooth_trans


def convert_motion_smpl_to_kimodo(smpl_motion: Tensor, admm_margin: float = 0.06) -> Tensor:
    """
    SMPL Root 198-dim → KIMODO Root 198-dim 在线转换.
    
    步骤:
    1. [0:3] raw_trans → smooth_trans (ADMM XZ 平滑)
    2. [3:135] rot6d 透传
    3. [135:198] pos_rel_raw_pelvis → pos_rel_smooth_root
       pos_new[j] = pos_old[j] + (raw_trans - smooth_trans)  (参考系平移)
    """
    raw_trans = smpl_motion[..., 0:3]           # (B, T, 3)
    rot6d = smpl_motion[..., 3:135]             # (B, T, 132) — 透传
    pos_rel_pelvis = smpl_motion[..., 135:198]  # (B, T, 63)
    
    smooth_trans = smpl_trans_to_smooth_trans(raw_trans, admm_margin)
    
    # Position 参考系转换: pelvis → smooth_root
    # FK(rot)[j] - raw_pelvis + raw_pelvis - smooth_trans
    # = pos_rel_pelvis + (raw_trans - smooth_trans) 的 position 分量
    trans_offset = (raw_trans - smooth_trans).unsqueeze(-2).expand_as(
        pos_rel_pelvis.view(*pos_rel_pelvis.shape[:-1], 21, 3)
    ).reshape_as(pos_rel_pelvis)  # broadcast to 21 joints
    pos_rel_smooth = pos_rel_pelvis + trans_offset
    
    return torch.cat([smooth_trans, rot6d, pos_rel_smooth], dim=-1)  # (B, T, 198)
```

### D.4 测试规范

```python
def test_online_conversion():
    """
    单元测试: SMPL 198 → KIMODO 198 转换正确性验证。
    
    验收标准:
    - smooth_trans XZ 平滑度 > raw_trans XZ 平滑度 (jerk 更小)
    - smooth_trans Y == raw_trans Y (exact)
    - rot6d: 完全不变 (零误差)
    - pos_rel_smooth 和 pos_rel_pelvis 的差值 == (raw_trans - smooth_trans) broadcast
    - smooth_trans 与 raw_trans 的最大偏移 ≤ admm_margin (6cm)
    """
    smpl_motion = random_smpl_motion(B=4, T=196, dim=198)
    kimodo_motion = convert_motion_smpl_to_kimodo(smpl_motion)
    
    # 维度不变
    assert kimodo_motion.shape == smpl_motion.shape  # (B, T, 198)
    
    # Y 轴保持不变
    assert (kimodo_motion[..., 1] - smpl_motion[..., 1]).abs().max() == 0
    
    # Rotation 完全透传
    assert (kimodo_motion[..., 3:135] - smpl_motion[..., 3:135]).abs().max() == 0
    
    # XZ 偏移在 margin 内
    xz_offset = (kimodo_motion[..., [0, 2]] - smpl_motion[..., [0, 2]]).abs().max()
    assert xz_offset <= 0.06 + 1e-6
```

---

*文档结束。v2.0 核心变化：用 STP (Semantic-Temporal Planning，语义时空规划) 替换 SEAT——模型先推理"文本内容对应动作的什么时间、什么部位，哪些已被 motion condition 给定，哪些需要从 text 规划"，再执行生成。STP 三组件: (a) GTA 接地文本增强——LLM 标注 caption 的时空对应关系；(b) CAPT 条件感知规划标记——轻量 planning head (<0.01% 参数) 预测语义-时空规划图；(c) PGG 规划引导生成——将规划图反馈到 CPOS 推理 schedule 实现空间自适应引导。与任务类型无关 (T2M/M2M/inpainting/editing 统一适用)。之前版本变化保留: v1.9 修正 §4.3 MAN+no_inactive 3通道594-dim、CPOS alpha replacement guidance 机制。*
