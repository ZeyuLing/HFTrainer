# HyMotion M2M 下一代方案：条件解耦编排流匹配 (CDO-FM)

**Condition-Decoupled Orchestration Flow Matching**

文档版本: 1.7 | 日期: 2026-05-12 | 状态: 方案设计 (v1.7: 删除 TCC/CCFM/PCE，简化 KIMODO Root 为 198-dim smooth traj 替换，在线转换，V3 sampler 统一，loss 权重待定)

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
| D2 | **Translation 信号占比过低** (10.2% vs KIMODO 40.5%) | P1 | `trans_dim_weight=5.0` 但仅 3/135 维度 | 提高 `trans_dim_weight` 或 KIMODO Root (§7) | ✅ 纳入 (via §7) |
| D4 | **Local rotation 误差沿运动链放大** | P1 | 无 FK 约束时固有问题 | 启用 FK loss (D1) 即可缓解 | ✅ 纳入 (via D1) |

> **已确认不纳入 Phase 0 的方案**:
> - ~~D3: Translation augmentation (`transl_aug_prob`)~~: KIMODO Root 方案使用 ADMM 平滑替代 augmentation，效果更佳且更合理。SMPL Root 版本暂不启用 transl_aug，以保持两版实验的纯对比。
> - ~~D5: Foot contact / ground constraint 监督~~: 需要新增 foot contact 通道（扩展运动表征维度），复杂度高，延后到 Phase 3 (§6 CCFM)。
> - ~~TCC (Typed Condition Canvas)~~: 架构改动大，延后到 Phase 2。

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
| **E-T1: ADMM 平滑 translation** | KIMODO Root (§7): 对 [0:3] translation 施加 ADMM 平滑 (margin=6cm) | 减少 translation 高频噪声，改善滑步 | 低（预处理 + 重训） |
| **E-T2: 显式 velocity 通道** | 扩展 198-dim → 261-dim: 增加 21×3 joint velocity [198:261] | 改善时间连续性和滑步检测 | 中（改 representation + 重训），**延后到后续实验** |

> **已移除的实验**：
> - ~~E-T3: Translation augmentation~~: 由 KIMODO Root ADMM 平滑替代
> - ~~E-T4: Foot contact 通道~~: 延后到 Phase 3 (CCFM)

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

> 通过**密度感知路由机制**动态编排语义层（text）与结构层（motion condition）的相对贡献，使单一模型自适应地在纯文本生成与强条件补全之间平滑插值，同时引入 KIMODO 风格的 smooth root trajectory 提升生成质量。

### 3.3 核心创新点

1. **Density-Modulated Dual-Stream Attention (DM-DSA)**: 基于条件密度的双流注意力门控，text stream 和 motion condition stream 的贡献由可学习的密度调制器动态平衡

2. **Dual Root Representation + Loss Alignment**: 两套 root 表征方案 (SMPL raw / KIMODO smooth trajectory) 的 A/B 实验，统一 position loss 在 relative-to-root 空间计算，移除 t² weighting

3. **Unified V3 Condition Sampler**: Rank-K Boolean Tensor Prior，统一 caption/uncond configs 的 motion condition 训练采样

4. **Decoupled CFG**: 二级解耦 classifier-free guidance，独立控制 text 和 motion condition 的引导强度

---

## 4. 架构设计

### 4.1 整体架构

```
                         ┌─────────────────────────────────────┐
                         │          CDO-FM Architecture         │
                         └─────────────────────────────────────┘

  Text Input                                          Motion Condition Input
  "a person walks                                     [VACE: inactive + reactive + mask]
   forward and waves"                                 [sparse keyframes / trajectory / ...]
       │                                                      │
       ▼                                                      ▼
  ┌─────────────┐                                    ┌──────────────────┐
  │ Text Encoder │ (frozen Qwen3 + CLIP-L)           │ VACE Condition   │
  │             │                                    │ Encoder (现有)    │
  └──────┬──────┘                                    └────────┬─────────┘
         │                                           cond (B,L,198)
   ctxt (B,S,4096)                                   mask (B,L,198)
   vtxt (B,1,768)                                    density (B,L)
         │                                                  │
         │              ┌──────────────┐                    │
         │              │   Density    │◄───────────────────┘
         │              │  Modulator   │
         │              │   (DM)       │
         │              └──────┬───────┘
         │                     │ gate_text (B,1), gate_motion (B,L)
         │                     │
         ▼                     ▼
  ╔══════════════════════════════════════════════════════╗
  ║            MMDiT Backbone (reused)                   ║
  ║                                                      ║
  ║  ┌──────────────────────────────────────────────┐   ║
  ║  │  Dual-Stream Blocks (×N_double)              │   ║
  ║  │                                              │   ║
  ║  │  Motion Stream:                              │   ║
  ║  │    x_t + VACE features → self-attn → FFN     │   ║
  ║  │                                              │   ║
  ║  │  Text Stream:                                │   ║
  ║  │    ctxt → self-attn → FFN                    │   ║
  ║  │                                              │   ║
  ║  │  Cross-stream: gated joint attention          │   ║
  ║  │    gate_text · text_attn + gate_motion ·     │   ║
  ║  │    motion_attn                               │   ║
  ║  └──────────────────────────────────────────────┘   ║
  ║                                                      ║
  ║  ┌──────────────────────────────────────────────┐   ║
  ║  │  Single-Stream Blocks (×N_single)            │   ║
  ║  │    [motion_tokens; text_tokens] → attn → FFN │   ║
  ║  └──────────────────────────────────────────────┘   ║
  ║                                                      ║
  ╚══════════════════════════════════════════════════════╝
         │
         ▼
  ┌─────────────────┐
  │  Flow Velocity  │
  │  Prediction     │
  │  v_θ (B,L,198)  │
  └─────────────────┘
         │
         ▼
  ┌─────────────┐
  │  ODE Solve  │
  │  + Replace  │
  │  Guidance   │
  └─────────────┘
         │
         ▼
    Output Motion (B, L, 198)
```

### 4.2 模块详细设计

#### 4.2.1 Density-Modulated Dual-Stream Attention (DM-DSA)

**动机**: Text 和 motion condition 的信息密度天然不对称。当 motion condition 很丰富（如 dense keyframes）时，text 的边际信息量很低，但当 motion condition 很稀疏时，text 是唯一的语义指导。需要一个**自适应机制**来平衡两者。

**设计**:

```python
class DensityModulator(nn.Module):
    """
    基于条件密度计算 text/motion condition 的贡献权重。
    密度高 → 增强 motion stream、适当保留 text stream
    密度低 → 增强 text stream

    关键：不是简单的 hard routing，而是 soft gating，
    确保即使在高密度条件下 text 仍有一定影响（语义编辑需要）。
    """
    def __init__(self, dim=1024, min_text_gate=0.15):
        super().__init__()
        self.min_text_gate = min_text_gate  # text 贡献的下限，防止完全忽略

        # 多粒度密度特征
        self.frame_density_enc = SinusoidalEncoding(dim=dim//2)  # per-frame 密度
        self.global_density_enc = SinusoidalEncoding(dim=dim//2)  # 全局密度

        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 2),  # [gate_text, gate_motion]
        )
        # 初始化为均衡门控
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, torch.tensor([0.5, 0.5]).log())

    def forward(self, condition_mask, timestep_emb):
        """
        Args:
            condition_mask: (B, L, 198) — 1=生成，0=已知
            timestep_emb:   (B, D) — 时间步嵌入（扩散早期需要更多语义指导）
        Returns:
            gate_text:   (B, 1) — text stream 权重
            gate_motion: (B, 1) — motion condition stream 权重
        """
        # 计算多尺度密度
        frame_density = condition_mask.mean(dim=-1)          # (B, L)  per-frame
        global_density = condition_mask.mean(dim=(-1, -2))   # (B,)    global

        # 编码密度
        frame_feat = self.frame_density_enc(frame_density)   # (B, L, D/2)
        global_feat = self.global_density_enc(global_density) # (B, D/2)

        # 结合时间步（扩散早期=高噪声，更需要语义指导）
        density_feat = torch.cat([global_feat, timestep_emb[:, :self.dim//2]], dim=-1)

        # 计算门控
        gates = self.gate_net(density_feat).softmax(dim=-1)  # (B, 2)
        gate_text = gates[:, 0:1].clamp(min=self.min_text_gate)
        gate_motion = gates[:, 1:2]

        # 归一化
        total = gate_text + gate_motion
        gate_text = gate_text / total
        gate_motion = gate_motion / total

        return gate_text, gate_motion
```

**在 DiT Block 中的应用**:

```python
class DualStreamDiTBlock(nn.Module):
    """修改后的 dual-stream block，集成密度调制"""

    def forward(self, x_motion, x_text, gate_text, gate_motion, ...):
        # 各自的 self-attention
        motion_out = self.motion_self_attn(x_motion)
        text_out = self.text_self_attn(x_text)

        # 关键改动：密度调制的 cross-stream attention
        # 原始: joint_attn = concat_attn([motion, text])
        # 新: gated contribution

        # Motion 对 Text 的 cross-attention（获取语义指导）
        m2t_attn = self.cross_attn_m2t(q=motion_out, kv=text_out)
        # Text 对 Motion 的 cross-attention（获取空间上下文）
        t2m_attn = self.cross_attn_t2m(q=text_out, kv=motion_out)

        # 密度调制融合
        motion_out = motion_out + gate_text * m2t_attn + gate_motion * self.motion_cond_proj(vace_features)
        text_out = text_out + t2m_attn

        # FFN
        motion_out = motion_out + self.motion_ffn(motion_out)
        text_out = text_out + self.text_ffn(text_out)

        return motion_out, text_out
```

**关键设计决策**:
- `min_text_gate=0.15`: 即使在最密集的条件下（如修复任务，只有少数帧待生成），text 至少保留 15% 的贡献。这确保了"语义编辑"类任务（如"让这个走路变得更高兴"）在高条件密度下仍然有效
- 门控权重与 timestep 耦合：扩散过程的早期（高噪声）更需要 text 的全局语义指导，后期（低噪声）更依赖 motion condition 的精确空间约束
- Soft gating 而非 hard routing：确保梯度可以流经两个 stream

### 4.3 输入张量格式

沿用当前 VACE 格式，不引入额外通道:

| 维度 | 当前 VACE | CDO-FM (proposed) | 变化 |
|------|-----------|-------------------|------|
| 噪声状态 | x_t (B,L,198) | x_t (B,L,198) | 不变 |
| 条件值 | inactive (B,L,198) | inactive (B,L,198) | 不变 |
| 反应通道 | reactive (B,L,198) | reactive (B,L,198) | 不变 |
| 掩码 | mask (B,L,198) | mask (B,L,198) | 不变 |
| **模型输入** | **(B,L,792)** → proj | **(B,L,792)** → proj | **不变** |

关键: 不改变输入格式，DM-DSA 在 attention 层面操作，不增加输入维度。

### 4.4 CFG (Classifier-Free Guidance) 改进

当前 CFG 只有一级（drop text），在 motion condition 存在时 text 的 CFG 效果极弱。

**提议：二级解耦 CFG (Decoupled CFG)**

```
v_uncond     = model(x_t, null_text, null_motion_cond)   # 完全无条件
v_motion     = model(x_t, null_text, motion_cond)        # 仅 motion
v_text       = model(x_t, text,      null_motion_cond)   # 仅 text
v_full       = model(x_t, text,      motion_cond)        # 全条件

# 解耦引导
v_guided = v_uncond
         + w_motion * (v_motion - v_uncond)       # motion 条件引导
         + w_text   * (v_full   - v_motion)       # text 的增量引导（在 motion 基础上）
```

**优势**:
- `w_text` 控制的是 text 在 motion condition 基础上的**增量贡献**，而非绝对贡献
- 当 motion condition 很强时，`v_motion` 已经很好，`v_full - v_motion` 的差异主要体现在 text 要求的**风格/情感**变化上
- 用户可以独立调节 `w_motion` 和 `w_text`

**训练适配**: 需要三种 dropout 组合
- (drop_text=True, drop_motion=True): 训练 `v_uncond`
- (drop_text=True, drop_motion=False): 训练 `v_motion`
- (drop_text=False, drop_motion=False): 训练 `v_full`
- (drop_text=False, drop_motion=True): 可选，训练 `v_text`

推荐采样概率: `p(both_drop)=0.05, p(text_drop)=0.10, p(motion_drop)=0.05, p(no_drop)=0.80`

---

## 5. 训练策略

### 5.1 Text-Motion Condition Dropout 协调

```python
# 训练时的 dropout schedule
# 关键: 确保 (有text, 有motion condition) 的组合占多数
dropout_schedule = {
    'text_only':     0.15,  # drop motion condition, keep text → 强制 text 学习
    'motion_only':   0.10,  # drop text, keep motion condition → 标准 M2M
    'both':          0.05,  # drop both → 无条件 baseline
    'full':          0.70,  # keep both → 目标状态
}
```

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
| 无 contact/foot height 条件类型 | P2 | CCFM (§6) 解决 |
| 无 velocity 条件类型 | P3 | 需要扩展 dimensional kind |

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

时间模式 (§8.2.1):
  all      → E8(loop), E10(part-level)
  empty    → E1(T2M)
  interval → E2(inbetween), E7(first-frame), E13(multi-prompt), E14(transition), E15(prepend)
  periodic → E3(keyframe), E4(end-effector), E5(trajectory), E6(foot)
  renewal  → E6(foot, 不规则接触)
  markov   → E9(repair, checker-driven segments)

维度模式 (§8.2.2):
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
| **E5: + Explicit Velocity Channel** | E3 > E1 (KIMODO root 有效) | 在版本 B 基础上增加 21×3 joint velocity 通道 | 60% |
| **E6: + Foot Contact Channel** | E1 滑步仍未解决 | 增加 4-dim foot contact 信号 (CCFM) | 55% |
| **E7: + ADMM-only Translation** | E3 ≤ E1 (KIMODO root 无效) | 仅在版本 A 上用 ADMM 平滑 translation，不改 root | 65% |
| **E8: + TCC** | E1/E3 完成 | 替换 VACE 为 Typed Condition Canvas | 50% |
| **E9: + DM-DSA** | E8 完成 | 加入密度调制双流注意力 | 45% |
| **E10: + PCE** | E2/E4 text 效果不佳 | 三阶段渐进训练 | 50% |

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
- **E-Contact**: 接触质量专项评估

---

## 11. 与前沿方法对比及新颖性分析

### 11.1 方法对比矩阵

| 特性 | 当前 M2M | VACE (Wan2.1) | OmniGen2 | Seedance 2.0 | Step1X-Edit | **CDO-FM (Ours)** |
|------|---------|---------------|----------|--------------|-------------|-------------------|
| 条件类型感知 | ✗ (3通道固定) | ✗ (VCU通用) | ✗ | ✗ | ✓ (MLLM理解) | **✓ (TCC 类型嵌入)** |
| Text-Motion 平衡 | ✗ (竞争) | ✗ | ✓ (双路径) | ✓ (DB-DiT) | ✓ (MLLM路由) | **✓ (DM-DSA 密度调制)** |
| 解耦 CFG | ✗ | ✗ | ✗ | ? | ✗ | **✓ (二级解耦)** |
| 渐进训练 | ✗ | ✗ | ✓ (3阶段) | ? | ✓ (2阶段) | **✓ (PCE 3阶段)** |
| 物理约束 | ✗ | ✗ | ✗ | ✓ (physics-aware) | ✗ | **✓ (CCFM)** |
| 任务统一 | 部分 | ✓ | ✓ | ✓ | 编辑为主 | **✓ (全覆盖)** |
| 运动领域适配 | ✓ | ✗ (视频) | ✗ (图像) | ✗ (视频) | ✗ (图像) | **✓** |

### 11.2 新颖性论证

1. **Density-Modulated Dual-Stream Attention**: 不同于 UniCombine 的 LoRA switching 或 OmniGen2 的 hard dual-path，DM-DSA 实现了条件密度驱动的 soft routing，这是首次在 motion generation 中提出。其核心洞察——条件密度的变化需要动态调整语义（text）vs 结构（motion）信号的相对权重——具有普适性，可推广到 video/image inpainting。

2. **Typed Condition Canvas**: 不同于 VACE 的无类型 3 通道编码，TCC 赋予每个空间条件明确的语义类型。这使得模型能够用不同的策略处理 keyframe 约束（高置信度精确约束）和 edit source（低置信度参考信号），类似于 VACE 的 Concept Decoupling 但更精细。

3. **Contact-Conditioned Flow Matching**: 将物理接触状态集成到 flow matching 训练中，是首次在扩散式 motion generation 框架中实现端到端的接地感知生成（而非后处理）。

4. **Progressive Condition Exposure**: 受 LLM 指令微调中课程学习的启发，PCE 首次系统化地解决 multi-modal multi-task motion generation 中的 shortcut learning 问题。

### 11.3 与 VACE 的关系

CDO-FM 可以视为 VACE 框架在 motion generation 领域的**深度进化**:
- VACE 提出了 `V=[T;F;M]` 的统一编码 → 我们的 TCC 是其类型化扩展
- VACE 的 Context Adapter → 我们的 DM-DSA 是其密度感知版本
- VACE 的 Concept Decoupling → 我们的 TCC 条件类型实现了更细粒度的解耦

---

## 12. 顶会论文定位

### 12.1 推荐标题

**"CDO-FM: Condition-Decoupled Orchestration for Unified Text-and-Motion Conditioned Human Motion Generation"**

或更简洁:

**"MotionCanvas: Density-Aware Condition Orchestration for Universal Motion Generation"**

### 12.2 故事线

> 现有 motion generation 方法要么只做 text-to-motion (T2M)，要么只做 motion completion (M2M)，无法在一个模型中同时理解文本语义和空间运动约束。我们发现根因在于两类条件的**信息密度天然不对称**——一句话的信息量远低于 10 帧 dense keyframe。简单地将两者混入同一 conditioning pipeline 会导致模型学到忽略文本的 shortcut。
>
> 为此，我们提出 CDO-FM，通过 (1) 密度调制双流注意力实现 text/motion 条件的自适应平衡，(2) 类型化条件画布让模型区分不同语义的空间约束，(3) 渐进式条件暴露训练策略防止 shortcut learning，(4) 接触条件化 flow matching 实现端到端的物理感知生成。
>
> 在 XXX benchmark 上，CDO-FM 首次在单一模型中同时达到 T2M SOTA 和 M2M SOTA，且在 text-conditioned motion completion 这一新任务上显著优于所有 baseline。

### 12.3 目标会议

- **首选**: CVPR 2027 (DDL ~Nov 2026) / ICLR 2027 (DDL ~Oct 2026)
- **备选**: NeurIPS 2027 (DDL ~May 2027) / ECCV 2027 (DDL ~Mar 2027)

### 12.4 可能的审稿人关注点

| 审稿人质疑 | 预备回应 |
|-----------|---------|
| DM-DSA 是否过于复杂？简单 concat 是否足够？ | A2 消融实验直接对比 |
| TCC 的条件类型标注是否引入额外成本？ | 完全自动生成，零人工成本 |
| PCE 的阶段划分是否需要精细调参？ | 提供 sensitivity analysis |
| Contact loss 与现有 post-processing 的对比？ | A5 消融 + 后处理消融 |
| 135-dim motion representation 是否限制了方法通用性？ | 讨论扩展到 SMPL-X/手部 |

---

## 13. 风险与备选方案

### 13.1 风险评估

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|---------|
| DM-DSA 训练不稳定 | 中 | 高 | 渐进引入 gate (先固定 0.5/0.5，再放开学习) |
| TCC 条件类型对性能提升有限 | 低-中 | 中 | 退化为标准 VACE (type_embed=0)，不影响其他组件 |
| PCE 三阶段训练成本过高 | 中 | 中 | Phase 1 可直接从 T2M checkpoint 初始化跳过 |
| Contact label 提取不准确 | 低 | 低 | 使用多阈值 + 时间窗平滑提高鲁棒性 |
| Decoupled CFG 推理开销 3x | 高 | 中 | 提供 standard CFG 回退；或 distillation 减少步数 |

### 14.2 降级方案

如果完整 CDO-FM 的某些组件效果不理想，可以独立使用其中任一子系统：

**最小可行方案 (MVP)**: 修复 B4 (cond_mask_prob) + PCE 训练策略 + Decoupled CFG
- 预期: 解决 text conditioning 失效的 80% 问题
- 成本: 最低，只需改训练配置和推理代码
- 时间: 1-2 周

**中间方案**: MVP + DM-DSA
- 预期: 进一步平衡 text/motion，达到论文水平
- 成本: 需要修改 DiT block，重新训练
- 时间: 3-4 周

**完整方案**: 全部 CDO-FM 组件
- 预期: 最佳性能，完整的论文故事
- 成本: 全面架构升级
- 时间: 6-8 周

---

## 15. 实施路线图 [v1.5 更新, v1.6 修正]

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
□ Loss 对齐 (§7.4, §7.5):
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

### Phase 1: MVP — 训练策略改进 (Week 3-4)

```
□ 实现 PCE 三阶段训练 scheduler
□ 实现 Decoupled CFG (推理端)
□ 实现 TAL-v2 (对比损失)
□ Caption configs 从 v2 升级到 v3 sampler
□ 切换训练数据到 high_quality.json (456K)
□ 根据 E1-E4 结果决定后续实验方向
□ 评估: 确认 text conditioning 恢复工作
```

### Phase 2: 架构升级 — TCC + DM-DSA (Week 5-7)

```
□ 实现 TypedConditionCanvas
□ 修改 PrepareM2MCondition 添加条件类型自动标注
□ 实现 DensityModulator
□ 修改 DualStreamDiTBlock 集成 DM-DSA
□ 训练 E8 + E9 消融
```

### Phase 3: 质量增强 — CCFM (Week 7-9)

```
□ 实现 contact label 提取
□ 预处理: 对训练集生成 contact labels
□ 实现 ContactPredictionHead + contact_aware_loss
□ 实现 IK foot lock 后处理
□ 训练 E6: CCFM 消融
□ 训练最终版本
```

### Phase 4: 论文撰写 (Week 9-12)

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

### CDO-FM 架构升级（待实施）

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `hftrainer/models/motion/hymotion_m2m/bundle.py` | 修改 | 增加 TCC, ContactHead |
| `hftrainer/models/motion/hymotion_m2m/network/condition_routing.py` | 新增 | TypedConditionCanvas, DensityModulator |
| `hftrainer/models/motion/hymotion_m2m/network/hymotion_mmdit.py` | 修改 | DualStreamDiTBlock 集成 DM-DSA |
| `hftrainer/trainers/motion/hymotion_m2m_cdofm_trainer.py` | 新增 | CDO-FM Trainer (PCE + TAL-v2 + contact loss) |
| `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` | 修改 | Decoupled CFG + contact-guided IK |
| `hftrainer/datasets/motion/transforms/` | 修改 | PrepareM2MCondition 增加类型标注 + contact extraction |
| `configs/hymotion_m2m/cdofm/` | 新增 | CDO-FM 各阶段训练配置 |

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

## 附录 D: KIMODO Root 转换规范 [v1.6 重写]

### D.1 维度映射

```
SMPL Root (135-dim) → KIMODO Root (138-dim) 转换:

SMPL:                          KIMODO:
[0:3]   abs_trans    ──┬──→   [0:3]    smooth_trans = ADMM(abs_trans.xz) + abs_trans.y
                       └──→   [135:138] trans_residual = abs_trans - smooth_trans
[3:9]   root_rot6d   ──→     [3:9]    root_rot6d  (透传，不变)
[9:135]  body_rot     ──→     [9:135]  body_rot    (透传，不变)

逆转换 (KIMODO 138 → SMPL 135):
[0:3]    smooth_trans   ─┐
[135:138] trans_residual ─┴──→  [0:3]   abs_trans = smooth_trans + trans_residual
[3:9]    root_rot6d     ──→    [3:9]   root_rot6d  (透传)
[9:135]  body_rot       ──→    [9:135]  body_rot    (透传)

扩展版:
SMPL (198-dim) → KIMODO (201-dim):
[135:198] pos_rel_pelvis ──→  [138:201] pos_rel_smooth_root (参考系: smooth_trans 替代 raw_trans)
```

**关键**: rotation 部分完全不变，仅 translation 做 smooth + residual 拆分。转换精确可逆，零信息丢失。

### D.2 ADMM 平滑参数

| 参数 | 值 | 说明 |
|------|-----|------|
| margin | 0.06m (6cm) | 平滑 XZ 轨迹的最大偏移约束 |
| step_size | 0.25 × sqrt(diag_max) | 自适应步长 |
| iterations | 100 per level | 每层优化迭代数 |
| smoothed_axes | XZ only | Y 轴高度保持原始值 |

### D.3 转换函数实现

```python
def smpl_root_to_kimodo_root(smpl_motion: Tensor, admm_margin: float = 0.06) -> Tensor:
    """
    SMPL 135-dim → KIMODO 138-dim.
    Rotation 透传，translation 做 ADMM 平滑 + residual 拆分。
    """
    raw_trans = smpl_motion[..., 0:3]       # (B, T, 3)
    root_rot6d = smpl_motion[..., 3:9]      # (B, T, 6) — 透传
    body_rot = smpl_motion[..., 9:135]      # (B, T, 126) — 透传
    
    smooth_trans = admm_smooth_xz(raw_trans, margin=admm_margin)  # XZ 平滑, Y 透传
    trans_residual = raw_trans - smooth_trans                      # 残差
    
    return torch.cat([smooth_trans, root_rot6d, body_rot, trans_residual], dim=-1)  # (B, T, 138)


def kimodo_root_to_smpl_root(kimodo_motion: Tensor) -> Tensor:
    """
    KIMODO 138-dim → SMPL 135-dim.
    逆转换: raw_trans = smooth_trans + residual, rotation 透传。
    """
    smooth_trans = kimodo_motion[..., 0:3]        # (B, T, 3)
    root_rot6d = kimodo_motion[..., 3:9]          # (B, T, 6) — 透传
    body_rot = kimodo_motion[..., 9:135]          # (B, T, 126) — 透传
    trans_residual = kimodo_motion[..., 135:138]  # (B, T, 3)
    
    raw_trans = smooth_trans + trans_residual
    return torch.cat([raw_trans, root_rot6d, body_rot], dim=-1)  # (B, T, 135)
```

### D.4 可逆性测试规范

```python
def test_roundtrip_conversion():
    """
    单元测试: SMPL → KIMODO → SMPL roundtrip 精度验证。
    
    验收标准:
    - Translation: max_error < 1e-6 (float32 精度，仅浮点舍入误差)
    - Root rotation: exact (零误差，透传不变)
    - Body rotation: exact (零误差，透传不变)
    
    由于 rotation 完全透传、translation 仅做 smooth + residual 的可逆拆分，
    roundtrip 误差理论上仅来自浮点舍入，应接近零。
    """
    # 生成随机 SMPL motion
    smpl_motion = random_smpl_motion(B=4, T=196, dim=135)
    
    # Roundtrip
    kimodo_motion = smpl_root_to_kimodo_root(smpl_motion)
    recovered_smpl = kimodo_root_to_smpl_root(kimodo_motion)
    
    # 验证 translation (smooth + residual = original)
    assert (smpl_motion[..., 0:3] - recovered_smpl[..., 0:3]).abs().max() < 1e-6
    
    # 验证 root rotation (exact pass-through)
    assert (smpl_motion[..., 3:9] - recovered_smpl[..., 3:9]).abs().max() == 0
    
    # 验证 body rotation (exact pass-through)
    assert (smpl_motion[..., 9:135] - recovered_smpl[..., 9:135]).abs().max() == 0


def test_position_channels():
    """
    验证 position channels 在不同参考系下的正确性。
    
    SMPL 198: pos[j] = FK(rot)[j] - pelvis_pos  (joints 1-21)
    KIMODO 201: pos[j] = FK(rot)[j] - smooth_trans  (joints 1-21)
    
    由于 rotation 相同，FK 输出相同，区别仅在参考系原点:
    kimodo_pos[j] = smpl_pos[j] + (pelvis_pos - smooth_trans)
                  = smpl_pos[j] + trans_residual
    """
    pass  # 实现时补充
```

---

*文档结束。v1.7 核心简化：KIMODO Root 从 201-dim (smooth_trans + rot6d + body_rot + trans_residual) 简化为 198-dim (仅替换 translation [0:3] 为 ADMM平滑值，rotation/position 透传)，维度不变、架构无改动、在线转换即可。移除 TCC/CCFM/PCE 复杂方案，聚焦 Phase 0 首批实验：E1-E4 验证 SMPL Root vs KIMODO Root 的滑步改善效果。附录 D 中的 201-dim 转换规范已被 v1.7 简化超越，仅作历史参考。方案长期目标保持不变：通过密度感知的条件解耦编排 (CDO-FM)，实现 text 和 motion condition 的自适应平衡，并以接触条件化 flow matching 提升生成质量。*
