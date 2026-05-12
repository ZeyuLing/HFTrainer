# HyMotion M2M 下一代方案：条件解耦编排流匹配 (CDO-FM)

**Condition-Decoupled Orchestration Flow Matching**

文档版本: 1.5 | 日期: 2026-05-12 | 状态: 方案设计 (Phase 0: v2 caption bugs 已修复; 新增双 Root 表征方案 + KIMODO 对齐 + 条件采样分析 + 优先级实验计划)

---

## 目录

1. [问题诊断](#1-问题诊断)
2. [设计目标与约束](#2-设计目标与约束)
3. [核心方案概述](#3-核心方案概述)
4. [架构设计](#4-架构设计)
5. [训练策略](#5-训练策略)
6. [质量增强：接地感知生成](#6-质量增强接地感知生成)
7. [双 Root 表征方案：SMPL Root vs KIMODO Root](#7-双-root-表征方案smpl-root-vs-kimodo-root) **[v1.5 新增]**
8. [Motion Condition 训练采样分析](#8-motion-condition-训练采样分析) **[v1.5 新增]**
9. [实验计划（按优先级排序）](#9-实验计划按优先级排序) **[v1.5 重写]**
10. [评估指标与任务覆盖](#10-评估指标与任务覆盖) **[v1.5 从原 §7.2/§7.3 迁移]**
11. [与前沿方法对比及新颖性分析](#11-与前沿方法对比及新颖性分析)
12. [顶会论文定位](#12-顶会论文定位)
13. [风险与备选方案](#13-风险与备选方案)
14. [实施路线图](#14-实施路线图) **[v1.5 更新]**

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

评测显示 HyMotion M2M 相比 KIMODO 的最大质量差距是**滑步（foot skating）**。经过对 loss 计算、数据加载、translation augmentation 等环节的完整代码审计，确认**不是实现 bug，而是 5 项设计缺陷**：

| # | 缺陷 | 严重度 | 当前状态 | 修复方式 |
|---|------|--------|---------|---------|
| D1 | **FK keypoint loss 已实现但被禁用** (`keypoints3d_weight=0.0`) | P0 | 代码存在，config 关闭 | Config: 设为 10+ |
| D2 | **Translation 信号占比过低** (10.2% vs KIMODO 40.5%) | P1 | `trans_dim_weight=5.0` 但仅 3/135 维度 | 提高 `trans_dim_weight` 或增大 translation 维度权重 |
| D3 | **Translation augmentation 被禁用** (`transl_aug_prob=0.0`) | P2 | Config 关闭 | Config: 设为 0.5 |
| D4 | **Local rotation 误差沿运动链放大** | P1 | 无 FK 约束时固有问题 | 启用 FK loss (D1) 即可缓解 |
| D5 | **无 foot contact / ground constraint 监督** | P2 | 未实现 | 需新增 CCFM (§6) |

**代码验证**：translation-body motion 耦合实现正确。`load_smplx.py` 中 `process_transl()` 对 translation 和 root orientation 做一致的旋转增强；loss 计算中 translation (dims 0-3) 和 rotation (dims 3-135) 均参与 velocity loss，无遗漏。

**立即可行的 config-only 修复**（不改代码）：
1. `keypoints3d_weight=10.0`：启用已实现的 FK keypoint loss
2. `transl_aug_prob=0.5`：启用 translation augmentation

**对 CDO-FM 方案的影响**：D1-D3 的修复应纳入 Phase 0 baseline 重训；D5 由 §6 CCFM 解决。

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

**实验消融计划**（纳入 Phase 0 或 Phase 1）：

| 实验 | 变更 | 预期效果 | 成本 |
|------|------|---------|------|
| **E-T1: ADMM 平滑 translation** | 数据预处理: 对 [0:3] translation 施加 ADMM 平滑 (margin=6cm)，重新计算 mean/std | 减少 translation 高频噪声，改善滑步 | 低（预处理 + 重训） |
| **E-T2: 显式 velocity 通道** | 扩展 198-dim → 261-dim: 增加 21×3 joint velocity [198:261] | 改善时间连续性和滑步检测 | 中（改 representation + 重训） |
| **E-T3: Translation augmentation** | `transl_aug_prob=0.5`, `transl_aug_yaw_deg=180`, `transl_aug_offset_std=(1,0,1)` | 减少 position/orientation 过拟合 | 低（config-only） |
| **E-T4: Foot contact 通道** | 扩展 +4-dim foot contact binary labels | 提供显式地面约束信号 | 中（数据预处理 + 改 representation） |

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
| R4 | 解决滑步等生成质量问题 | P1 |
| R5 | 方案具备顶会论文新颖性 | P1 |
| R6 | 实现可行，不引入过大额外计算量（< 1.5x 当前训练成本） | P2 |

### 2.2 非功能约束

- 沿用 MMDiT (dual-stream → single-stream) 架构骨架，最大化复用已有代码和预训练权重
- 保持 135-dim 运动表征和 SMPL-22 骨骼
- 训练规模 ≤ 32 GPU × V100 (80GB)
- 与现有 eval pipeline (E1-E15) 完全兼容

---

## 3. 核心方案概述

### 3.1 方案名称

**条件解耦编排流匹配 (Condition-Decoupled Orchestration Flow Matching, CDO-FM)**

### 3.2 一句话描述

> 将扩散过程的条件注入**分层解耦**为语义层（text → global motion intent）、结构层（motion condition → spatial-temporal constraints）和质量层（physics → contact-aware refinement），通过**密度感知路由机制**动态编排各层的相对贡献，使单一模型自适应地在纯文本生成与强条件补全之间平滑插值。

### 3.3 核心创新点

1. **Density-Modulated Dual-Stream Attention (DM-DSA)**: 基于条件密度的双流注意力门控，text stream 和 motion condition stream 的贡献由可学习的密度调制器动态平衡

2. **Typed Condition Canvas (TCC)**: 替代固定的 inactive/reactive/mask 三通道，引入条件类型嵌入（type embedding），让模型区分 keyframe / trajectory / joint constraint / boundary 等不同语义的空间条件

3. **Contact-Conditioned Flow Matching (CCFM)**: 将地面接触状态作为显式条件通道，训练模型在生成过程中感知并尊重脚-地面物理约束

4. **Progressive Condition Exposure (PCE)**: 三阶段渐进式训练策略，从纯 T2M 逐步引入 motion condition，防止 text 信号在 multi-task 训练中被淹没

---

## 4. 架构设计

### 4.1 整体架构

```
                         ┌─────────────────────────────────────┐
                         │          CDO-FM Architecture         │
                         └─────────────────────────────────────┘

  Text Input                                          Motion Condition Input
  "a person walks                                     [sparse keyframes + mask]
   forward and waves"                                 [trajectory + joint constraints]
       │                                                      │
       ▼                                                      ▼
  ┌─────────────┐                                    ┌──────────────────┐
  │ Text Encoder │ (frozen Qwen3 + CLIP-L)           │ Typed Condition  │
  │             │                                    │ Canvas Encoder   │
  └──────┬──────┘                                    │   (TCC-Enc)      │
         │                                           └────────┬─────────┘
   ctxt (B,S,4096)                                      tcc (B,L,D_cond)
   vtxt (B,1,768)                                    cond_type (B,L,N_types)
         │                                           density (B,L,J)
         │                                                  │
         │              ┌──────────────┐                    │
         │              │   Density    │◄───────────────────┘
         │              │  Modulator   │
         │              │   (DM)       │
         │              └──────┬───────┘
         │                     │ gate_text (B,1), gate_motion (B,L,J)
         │                     │
         ▼                     ▼
  ╔══════════════════════════════════════════════════════╗
  ║            MMDiT Backbone (reused)                   ║
  ║                                                      ║
  ║  ┌──────────────────────────────────────────────┐   ║
  ║  │  Dual-Stream Blocks (×N_double)              │   ║
  ║  │                                              │   ║
  ║  │  Motion Stream:                              │   ║
  ║  │    x_t + TCC features → self-attn → FFN     │   ║
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
  ┌─────────────────┐     ┌──────────────────┐
  │  Flow Velocity  │     │  Contact Head    │
  │  Prediction     │     │  (auxiliary)     │
  │  v_θ (B,L,135) │     │  c_θ (B,L,2)    │
  └─────────────────┘     └──────────────────┘
         │                        │
         ▼                        ▼
  ┌─────────────┐          ┌────────────┐
  │  ODE Solve  │          │  Contact   │
  │  + Replace  │◄─────────│  Guided    │
  │  Guidance   │          │  IK Refine │
  └─────────────┘          └────────────┘
         │
         ▼
    Output Motion (B, L, 135)
```

### 4.2 模块详细设计

#### 4.2.1 Typed Condition Canvas (TCC)

**动机**: 当前 VACE 的 `[inactive, reactive, mask]` 是一种无类型的编码——模型只知道"这里有值/没值"，不知道"这是一个关键帧约束还是一个轨迹约束"。不同类型的条件应该有不同的影响方式。

**设计**:

```python
class TypedConditionCanvas(nn.Module):
    """
    将各种类型的 motion condition 编码为统一的条件画布。
    每种条件类型有独立的 type embedding，让模型区分条件语义。
    """
    # 条件类型枚举
    COND_TYPES = {
        'none':        0,   # 无条件（待生成区域）
        'keyframe':    1,   # 关键帧（完整 pose）
        'trajectory':  2,   # 轨迹约束（仅 translation）
        'joint_pos':   3,   # 关节位置约束
        'joint_rot':   4,   # 关节旋转约束
        'boundary':    5,   # 边界帧（用于过渡/衔接）
        'edit_source': 6,   # 编辑任务的源动作（低质量/待修改）
        'contact':     7,   # 接触状态约束
    }

    def __init__(self, motion_dim=135, num_types=8, type_embed_dim=64):
        super().__init__()
        # 可学习的条件类型嵌入
        self.type_embedding = nn.Embedding(num_types, type_embed_dim)

        # 条件值编码器（替代原 VACE 的 inactive/reactive 拼接）
        self.value_encoder = nn.Linear(motion_dim, motion_dim)

        # 类型-值融合
        self.fusion = nn.Sequential(
            nn.Linear(motion_dim + type_embed_dim, motion_dim),
            nn.SiLU(),
            nn.Linear(motion_dim, motion_dim),
        )
        nn.init.zeros_(self.fusion[-1].weight)  # zero-init for residual safety

    def forward(self, condition_motion, condition_mask, condition_type_ids):
        """
        Args:
            condition_motion: (B, L, 135) — 条件区域的运动值（非条件区域为 0）
            condition_mask:   (B, L, 135) — 二值 mask，1=待生成，0=已知条件
            condition_type_ids: (B, L, J=23) — 每个 (frame, joint-group) 的类型 ID
        Returns:
            tcc_features: (B, L, 135) — 编码后的条件画布特征
        """
        # 编码条件值
        cond_val = self.value_encoder(condition_motion)  # (B, L, 135)

        # 获取类型嵌入并扩展到 per-dim
        type_emb = self.type_embedding(condition_type_ids)  # (B, L, 23, type_embed_dim)
        type_emb = expand_joint_group_to_dim(type_emb, D=135)  # (B, L, 135, type_embed_dim)

        # 融合值 + 类型
        fused = self.fusion(torch.cat([
            cond_val.unsqueeze(-1).expand_as(type_emb[..., :1]),  # value broadcast
            type_emb
        ], dim=-1))  # (B, L, 135)

        # mask=0 的区域有条件特征，mask=1 的区域为 learned null
        tcc_features = fused * (1 - condition_mask) + self.null_cond * condition_mask

        return tcc_features
```

**关键改进 vs 当前 VACE**:
- **类型感知**: 模型知道每个位置提供的是什么类型的约束
- **统一编码**: 不再区分 inactive/reactive，统一为 value + type
- **可扩展**: 新增条件类型只需添加 type ID，无需改架构
- **编辑支持**: `edit_source` type 明确标记需要修复的区域

#### 4.2.2 Density-Modulated Dual-Stream Attention (DM-DSA)

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
            condition_mask: (B, L, 135) — 1=生成，0=已知
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
        motion_out = motion_out + gate_text * m2t_attn + gate_motion * self.motion_cond_proj(tcc_features)
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

#### 4.2.3 条件感知 AdaLN (Condition-Aware AdaLN)

**动机**: 当前 AdaLN 只接收 timestep embedding + 可选的 CDE。我们将其升级为同时感知条件密度和条件类型分布。

```python
class ConditionAwareAdaLN(nn.Module):
    """
    AdaLN 调制，同时考虑：
    1. Timestep t
    2. 条件密度 ρ (全局 + per-frame)
    3. 条件类型分布 τ (每种类型的覆盖比例)
    4. Text 存在标记 (有/无 text condition)
    """
    def __init__(self, dim, num_cond_types=8):
        super().__init__()
        # 条件类型分布编码
        self.type_dist_proj = nn.Linear(num_cond_types, dim)
        # text 存在标记
        self.text_flag_proj = nn.Linear(1, dim)
        # 融合 → AdaLN 参数
        self.adaln_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim * 3, dim * 6),  # scale, shift for 3 sub-layers
        )

    def forward(self, timestep_emb, density_emb, type_distribution, has_text):
        """
        type_distribution: (B, num_types) — 每种条件类型占总条件的比例
        has_text: (B, 1) — 是否有 text condition
        """
        combined = timestep_emb + density_emb + \
                   self.type_dist_proj(type_distribution) + \
                   self.text_flag_proj(has_text.float())
        return self.adaln_proj(combined)
```

### 4.3 输入张量格式对比

| 维度 | 当前 VACE | CDO-FM (proposed) | 变化 |
|------|-----------|-------------------|------|
| 噪声状态 | x_t (B,L,135) | x_t (B,L,135) | 不变 |
| 条件值 | inactive (B,L,135) | tcc_value (B,L,135) | 语义相同，编码方式不同 |
| 反应通道 | reactive (B,L,135) | — (合并到 TCC) | **移除独立通道** |
| 掩码 | mask (B,L,135) | mask (B,L,135) | 不变 |
| 类型信息 | 无 | type_emb (B,L,135) | **新增** |
| 接触状态 | 无 | contact (B,L,2) | **新增** |
| **模型输入** | **(B,L,540)** → proj | **(B,L,540+2)** → proj | 接近不变 |

关键: TCC 将 inactive + reactive 统一为一个带类型信息的条件通道，总输入维度几乎不变（+2 用于 contact），不增加主干计算量。

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

### 5.1 Progressive Condition Exposure (PCE)

这是本方案最关键的训练创新。核心思想：**不要从一开始就让模型面对所有条件类型的组合，而是渐进地引入条件复杂度**。

```
Phase 1: Text Foundation (T2M Warmup)
├─ 目标: 建立稳固的 text → motion 理解
├─ 数据: 100% 纯 T2M (mask = all 1, no motion condition)
├─ 训练: 标准 flow matching + text CFG (drop_prob=0.1)
├─ 时长: ~30% 总训练 steps
├─ 初始化: 从 T2M 1.0-Lite 预训练权重开始
└─ 验证: T2M FID/R-Precision 达到 T2M baseline 水平

Phase 2: Condition Introduction (渐进引入 motion condition)
├─ 目标: 在保持 text 理解的同时学习 motion condition
├─ 数据: 线性增加 motion condition 比例
│   ├─ Step 0:    80% T2M + 20% sparse condition (M6 keyframe only)
│   ├─ Step T/3:  50% T2M + 50% mixed condition (M3+M6)
│   └─ Step 2T/3: 30% T2M + 70% full condition mix (M1-M7)
├─ 训练: flow matching + 解耦 CFG + text awareness loss (TAL)
├─ 关键: TAL 确保 text 信号不被 motion condition 淹没
├─ 时长: ~40% 总训练 steps
└─ 验证: T2M 指标不下降 + M2M 指标稳步上升

Phase 3: Full Multi-Task Mastery
├─ 目标: 全面提升所有任务性能
├─ 数据: 100% 全 mask pattern (Rank-K 采样器)
│   ├─ T2M (M5): 10-15% (维持 text 能力)
│   ├─ Condition mix (M1-M4, M6-M7): 70-80%
│   └─ Edit/Repair: 10-15%
├─ 训练: 全特性 (解耦 CFG + TAL + contact loss + quality filter)
├─ 时长: ~30% 总训练 steps
└─ 验证: 全部 E1-E15 评估任务
```

### 5.2 Anti-Shortcut Training Mechanisms

防止模型学到"有 motion condition 就忽略 text"的捷径：

#### 5.2.1 Text-Awareness Loss v2 (TAL-v2)

升级当前 TAL，增加更强的约束：

```python
def text_awareness_loss_v2(pred_with_text, pred_without_text, pred_with_wrong_text,
                            condition_mask, mask_density):
    """
    三重文本感知损失:
    1. 有 text vs 无 text 的差异应大于阈值 (原 TAL)
    2. 正确 text vs 错误 text 的差异应大于阈值 (新: 对比损失)
    3. 差异量级与 mask_density 负相关 (新: 密度自适应)
    """
    # 原始 TAL: 有 text 应与无 text 不同
    text_effect = (pred_with_text - pred_without_text).norm(dim=-1)
    min_effect = adaptive_threshold(mask_density)  # 密度越高，阈值越低但不为零
    tal_original = F.relu(min_effect - text_effect).mean()

    # 新增: 对比损失 — 正确 text 与错误 text 应产生不同结果
    correct_dist = (pred_with_text - pred_without_text).norm(dim=-1)
    wrong_dist = (pred_with_wrong_text - pred_without_text).norm(dim=-1)
    # 正确 text 的影响应大于错误 text
    tal_contrastive = F.relu(wrong_dist - correct_dist + margin).mean()

    return tal_original + 0.5 * tal_contrastive
```

#### 5.2.2 Text-Motion Condition Dropout 协调

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

### 5.3 数据策略

#### 5.3.1 质量过滤

从 549K 样本切换到 456K 高质量子集（`high_quality.json`），预期 +3-5% 质量提升。

#### 5.3.2 Text Augmentation

当前每条 motion 只有一个 caption。为了增强 text 理解的鲁棒性：
- 同一 motion 的多种描述（不同详细程度、不同重点）
- 通过 LLM 改写现有 caption（已有 `--use-rewritten` 支持）
- 负样本: 随机配对错误 caption 用于 TAL-v2 的对比损失

#### 5.3.3 Condition Type 标注

当前 mask 只有 0/1，不包含类型信息。TCC 需要类型标注：
- **自动生成**: 在 `PrepareM2MCondition` 中根据 mask pattern 生成器自动设置 type ID
- M5 (全 mask) → type=none
- M6 (keyframe) → type=keyframe
- M3 (时间连续) → type=boundary (边界帧)
- M4 (关节连续) → type=joint_rot
- Edit mode → type=edit_source
- 这不需要额外的人工标注

---

## 6. 质量增强：接地感知生成

### 6.1 Contact-Conditioned Flow Matching (CCFM)

**核心思想**: 将脚-地面接触状态作为显式条件，让模型在生成过程中直接"知道"哪些帧的脚应该落地。

#### 6.1.1 Contact Label 提取

从 GT motion 自动提取接触标签（训练时使用）：

```python
def extract_contact_labels(motion, skeleton, vel_thresh=0.01, height_thresh=0.03):
    """
    从 GT 动作中提取双脚接触标签。
    Args:
        motion: (T, 135) — 标准化前的原始动作
        skeleton: SMPL-22 骨骼定义
    Returns:
        contact: (T, 2) — [left_foot, right_foot] 接触状态 (0/1)
    """
    # 通过 FK 计算全局关节位置
    joint_positions = forward_kinematics(motion, skeleton)  # (T, 22, 3)

    # 脚踝关节索引 (SMPL-22: L_Ankle=7, R_Ankle=8)
    foot_pos = joint_positions[:, [7, 8], :]  # (T, 2, 3)
    foot_vel = torch.diff(foot_pos, dim=0, prepend=foot_pos[:1])  # (T, 2, 3)
    foot_speed = foot_vel.norm(dim=-1)  # (T, 2)
    foot_height = foot_pos[:, :, 1]  # (T, 2) — Y 轴为高度

    # 接触判定: 速度低 AND 高度低
    contact = (foot_speed < vel_thresh) & (foot_height < height_thresh)
    return contact.float()
```

#### 6.1.2 Contact 作为条件通道

```python
# 在 TCC 编码中追加 contact 通道
model_input = torch.cat([
    x_t,            # (B, L, 135) — 噪声状态
    tcc_features,   # (B, L, 135) — 类型化条件特征
    mask,           # (B, L, 135) — 生成 mask
    contact_cond,   # (B, L, 2)   — 双脚接触状态（新增）
], dim=-1)  # (B, L, 407) → 输入 projection
```

#### 6.1.3 Contact-Aware Loss

```python
def contact_aware_loss(pred_velocity, pred_motion, gt_contact, skeleton):
    """
    在预测接触的帧上惩罚脚部滑动。
    """
    # 通过 FK 获取脚部位置
    pred_foot_pos = fk_foot(pred_motion, skeleton)  # (B, T, 2, 3)
    pred_foot_vel = torch.diff(pred_foot_pos, dim=1)  # (B, T-1, 2, 3)

    # 接触帧的脚部速度应为零
    contact_mask = gt_contact[:, 1:, :]  # (B, T-1, 2)
    foot_skating_loss = (pred_foot_vel.norm(dim=-1) * contact_mask).mean()

    return foot_skating_loss
```

#### 6.1.4 Contact Head (辅助预测)

在推理时，如果用户未提供 contact 信息，模型需自行预测合理的接触模式：

```python
class ContactPredictionHead(nn.Module):
    """轻量级接触预测头，与主 velocity head 并行"""
    def __init__(self, dim, num_feet=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, num_feet),
        )

    def forward(self, motion_features):
        return self.proj(motion_features).sigmoid()
```

### 6.2 推理时 IK 精修

即使训练时有 contact loss，推理时仍可能有轻微滑步。增加一个轻量后处理：

```python
def ik_foot_lock(motion, predicted_contacts, skeleton, blend_frames=3):
    """
    对预测的接触帧执行 IK 锁脚。
    1. 找到每个接触段的第一帧，记录脚部位置
    2. 在接触段内，固定脚部位置
    3. IK 求解调整膝/踝关节旋转
    4. 在接触段边界做 blend_frames 帧的平滑过渡
    """
    # ... IK 实现
    return refined_motion
```

### 6.3 其他质量增强

#### 6.3.1 Velocity Consistency Loss

惩罚速度的不平滑变化（减少抖动）：

```python
vel_loss = smooth_l1(pred_velocity[:, 1:] - pred_velocity[:, :-1])
acc_loss = smooth_l1(pred_accel[:, 1:] - pred_accel[:, :-1])
```

#### 6.3.2 FK Consistency Loss

确保 rotation 和 position 的一致性：

```python
# 从预测的 rot6d 通过 FK 计算 position
pred_pos_from_rot = forward_kinematics(pred_rot6d, skeleton)
# 与预测的 position 对比
fk_loss = mse(pred_pos_from_rot, pred_joint_pos)
```

---

## 7. 双 Root 表征方案：SMPL Root vs KIMODO Root [v1.5 新增]

### 7.1 动机

KIMODO 在滑步抑制上显著优于 HyMotion，根因之一是其 **smooth root trajectory + explicit heading** 表征。为了验证这一假设并找到最优方案，我们实现两套 root 表征并做 A/B 对比实验。

### 7.2 两版 Root 表征定义

#### 7.2.1 版本 A: SMPL Root（当前实现）

```
135-dim layout:
  [0:3]   = absolute pelvis translation (world XYZ)
  [3:9]   = pelvis rotation (rot6d, parent-relative = world-frame for pelvis)
  [9:135]  = 21 body joint rotations (rot6d, parent-relative)

198-dim layout (with position channels):
  [0:135]   = 同上
  [135:198] = 21 × 3 joint positions (Scheme D: XZ relative to pelvis, Y absolute)
```

**特点**：
- translation 为原始 MoCap 数据，包含高频噪声/抖动
- root rotation 隐含了 heading 信息
- 与 SMPL forward kinematics 直接兼容，无需额外转换
- 总维度: 135 (core) / 198 (extended)

#### 7.2.2 版本 B: KIMODO Root（新实现）

```
134-dim layout (smooth root, 22 joints):
  [0:3]   = ADMM-smoothed pelvis translation (smooth XZ, raw Y)
  [3:5]   = global heading angle [cos(ψ), sin(ψ)]
  [5:131] = 21 body joint rotations (rot6d, parent-relative) — 与版本 A 相同
  [131:134] = translation residual (raw - smooth) — 用于可逆转换

197-dim layout (with position channels):
  [0:134]   = 同上
  [134:197] = 21 × 3 joint positions (Scheme D: XZ relative to smooth root, Y absolute)
```

> **注意**: 我们使用 SMPL-22 骨架（不是 KIMODO 的 SOMA-30），因此关节数与版本 A 一致。只替换 root 表征部分。

**关键设计决策**:

1. **Heading 从 root rotation 中解耦**: SMPL root rotation 的 rot6d 编码了完整 3D 旋转 (6 dims)，其中 heading (yaw) 仅占一个自由度。KIMODO 显式抽取 heading 为 2-dim `[cos(ψ), sin(ψ)]`，让 heading 信号更清晰。
2. **ADMM 平滑 translation**: 仅对 XZ 轴施加 ADMM 优化 (margin ≤ 6cm)，保留 Y 轴高度信息。去除 translation 的高频抖动。
3. **保留 translation residual**: 在最后 3 dims 存储 `raw_translation - smooth_translation`，确保转换完全可逆。
4. **Body rotation 不变**: dims [5:131] (版本 B) 与 dims [9:135] (版本 A) 完全相同——都是 21 个 body joint 的 parent-relative rot6d。差异仅在 root。
5. **Position channels 参考系切换**: position 从 relative to raw pelvis 改为 relative to smooth root，与 KIMODO 保持一致。

### 7.3 KIMODO Root → SMPL Root 转换（可逆性保证）

```python
def kimodo_root_to_smpl_root(kimodo_motion: Tensor) -> Tensor:
    """
    将 KIMODO root 表征转换回 SMPL root 表征。
    
    输入: (B, T, 134) KIMODO root layout
    输出: (B, T, 135) SMPL root layout
    
    转换步骤:
    1. 恢复原始 translation: raw_trans = smooth_trans + trans_residual
    2. 从 heading [cos(ψ), sin(ψ)] 重建 root rotation (仅 yaw):
       - heading_angle = atan2(sin_ψ, cos_ψ)
       - root_rotmat = Ry(heading_angle) — 纯 yaw 旋转
       - root_rot6d = rotmat_to_rot6d(root_rotmat)
       注意: KIMODO heading 仅编码 yaw，丢失了 pitch/roll。
       对于人体运动数据，pelvis 的 pitch/roll 通常很小，
       可以从 spine chain 的 body rotation 近似恢复。
    3. 拼接: [raw_trans(3), root_rot6d(6), body_rot(126)]
    """
    smooth_trans = kimodo_motion[..., 0:3]      # (B, T, 3)
    heading = kimodo_motion[..., 3:5]           # (B, T, 2) [cos, sin]
    body_rot = kimodo_motion[..., 5:131]        # (B, T, 126)
    trans_residual = kimodo_motion[..., 131:134] # (B, T, 3)
    
    # Step 1: 恢复原始 translation
    raw_trans = smooth_trans + trans_residual
    
    # Step 2: 从 heading 重建 root rotation
    heading_angle = torch.atan2(heading[..., 1], heading[..., 0])  # (B, T)
    root_rot6d = yaw_to_rot6d(heading_angle)  # (B, T, 6)
    
    # Step 3: 拼接
    smpl_motion = torch.cat([raw_trans, root_rot6d, body_rot], dim=-1)  # (B, T, 135)
    return smpl_motion

def smpl_root_to_kimodo_root(smpl_motion: Tensor, admm_margin: float = 0.06) -> Tensor:
    """
    将 SMPL root 表征转换为 KIMODO root 表征。
    
    输入: (B, T, 135) SMPL root layout
    输出: (B, T, 134) KIMODO root layout
    
    转换步骤:
    1. 提取 translation 和 root rotation
    2. ADMM 平滑 translation XZ (Y 保留)
    3. 从 root rotation 提取 heading angle (yaw)
    4. 计算 translation residual
    5. 拼接: [smooth_trans(3), heading(2), body_rot(126), trans_residual(3)]
    """
    raw_trans = smpl_motion[..., 0:3]
    root_rot6d = smpl_motion[..., 3:9]
    body_rot = smpl_motion[..., 9:135]
    
    # Step 1: ADMM 平滑 translation (XZ only)
    smooth_trans = admm_smooth_xz(raw_trans, margin=admm_margin)
    
    # Step 2: 提取 heading (yaw from root rotation)
    root_rotmat = rot6d_to_rotmat(root_rot6d)  # (B, T, 3, 3)
    heading_angle = extract_yaw(root_rotmat)     # (B, T)
    heading = torch.stack([torch.cos(heading_angle), 
                          torch.sin(heading_angle)], dim=-1)  # (B, T, 2)
    
    # Step 3: Translation residual
    trans_residual = raw_trans - smooth_trans     # (B, T, 3)
    
    # Step 4: 拼接
    kimodo_motion = torch.cat([smooth_trans, heading, body_rot, trans_residual], dim=-1)
    return kimodo_motion
```

**可逆性分析**:

| 方向 | 是否精确可逆 | 说明 |
|------|-------------|------|
| SMPL → KIMODO → SMPL (translation) | ✅ 精确 | 通过 `trans_residual` 完全恢复 |
| SMPL → KIMODO → SMPL (heading/yaw) | ✅ 精确 | `atan2(sin,cos)` → `Ry(θ)` 无信息丢失 |
| SMPL → KIMODO → SMPL (pitch/roll) | ⚠️ 近似 | KIMODO heading 仅编码 yaw，pitch/roll 信息丢失。对人体 pelvis (pitch/roll 通常 < 5°) 影响极小 |
| KIMODO → SMPL (推理输出) | ✅ 可行 | 推理后直接调用 `kimodo_root_to_smpl_root()` |

### 7.4 Loss 对齐：Position Loss 在 Relative-to-Root 空间计算

当前实现中 keypoint3d loss 已经在 relative to root 空间计算（参见 `m2m_loss.py:222`）:
```python
local_keypoints3d = pred_keypoints3d[:, :, 1:22] - pred_keypoints3d[:, :, 0:1, :]
```

但 198-dim 中的 position channels ([135:198]) 的 loss（x1 loss、velocity loss）在绝对空间计算，需要对齐:

**修改方案**:
1. **Position channels loss**: 对于版本 B (KIMODO root)，position channels 已经是 relative to smooth root，loss 自然在相对空间
2. **对于版本 A (SMPL root)**: 也改为在 relative to root 空间计算 position loss，与版本 B 统一
3. **实现**: 在 `m2m_loss.py` 中，计算 position loss 前先减去 root position：
   ```python
   # Before:
   pos_loss = smooth_l1(pred_x1[..., 135:198], target_x1[..., 135:198])
   
   # After (unified):
   pred_pos_rel = pred_x1[..., 135:198] - expand_to_joints(pred_x1[..., 0:3])
   target_pos_rel = target_x1[..., 135:198] - expand_to_joints(target_x1[..., 0:3])
   pos_loss = smooth_l1(pred_pos_rel, target_pos_rel)
   ```

### 7.5 Loss 对齐：移除 t² Timestep Weighting

当前 `kimodo_aux_loss.py` 中对辅助 loss 施加了 t² 加权:
```python
# kimodo_aux_loss.py line 280-283
if self.timestep_squared_weighting and timesteps is not None:
    t_sq = (timesteps.to(pred_world.device) ** 2)  # (B,)
    per_frame = per_frame * t_sq.unsqueeze(-1)
```

**t² 加权的效果**: `E[t²] = 1/3`（t~Uniform[0,1]），意味着靠近纯噪声端 (t→0) 的样本贡献被大幅降低。原始设计意图是：在纯噪声时 FK 计算无意义，避免辅助 loss 产生错误梯度。

**移除理由**:
1. **与 KIMODO 对齐**: KIMODO 原版不使用 t² 加权
2. **Flow matching 特性**: 在 rectified flow 中，velocity prediction 的目标 `v = x1 - x0` 与 t 无关，所有 t 的 loss 等权是合理的
3. **辅助 loss 在低 t 也有信号**: 即使 `x_t = (1-t)*noise + t*x1` 在 t 小时接近纯噪声，模型的**预测 x1** 仍然有意义（是模型对 clean motion 的最佳估计），FK 在预测 x1 上计算而非在 x_t 上
4. **实验对齐**: 移除 t² 使两版模型的 loss 完全可比

**修改**: 将 `timestep_squared_weighting=True` 改为 `timestep_squared_weighting=False`（config 修改，不改代码）。

### 7.6 两版配置对比

| 配置项 | 版本 A: SMPL Root | 版本 B: KIMODO Root |
|--------|------------------|-------------------|
| **motion_dim** | 198 (135 core + 63 pos) | 197 (134 core + 63 pos) |
| **root 表征** | [trans(3), root_rot6d(6)] = 9 dims | [smooth_trans(3), heading(2), residual(3)] = 8 dims |
| **body rotation** | [9:135] 21×6 rot6d | [5:131] 21×6 rot6d (同) |
| **position channels** | relative to raw pelvis | relative to smooth root |
| **position loss 空间** | relative to root (对齐后) | relative to smooth root (自然) |
| **t² weighting** | ❌ 移除 | ❌ 移除 |
| **velocity loss** | ✅ | ✅ |
| **FK keypoint loss** | keypoints3d_weight=10.0 | keypoints3d_weight=10.0 |
| **KIMODO aux losses** | aux_joint_pos=50, aux_joint_vel=500, aux_fk=1500 | 同 |
| **motion smoothness** | motion_smoothness_weight=0.5 | 同 |
| **trans_dim_weight** | 5.0 | 5.0 |
| **transl_aug** | transl_aug_prob=0.5 | 不需要 (ADMM 已平滑) |
| **ADMM 预处理** | 不需要 | 离线计算 + 缓存 |
| **推理后转换** | 不需要 | kimodo_root_to_smpl_root() |

### 7.7 数据预处理管线

版本 B (KIMODO Root) 需要离线预处理，将原始 MoCap 数据转换为 KIMODO root 格式:

```
输入: data/annotation/train_hymotion_400h.json → .npz files (135-dim)
    ↓
Step 1: 对每个 motion 的 translation [0:3] 执行 ADMM XZ 平滑
    ↓
Step 2: 从 root rotation [3:9] 提取 yaw heading
    ↓
Step 3: 计算 translation residual = raw - smooth
    ↓
Step 4: 重新计算 position channels (relative to smooth root)
    ↓
Step 5: 拼接为 197-dim，计算新的 mean/std 统计量
    ↓
输出: data/annotation/train_hymotion_400h_kimodo_root.json + .npz files (197-dim)
     + data/annotation/mean_std_197dim_kimodo_root.npz
```

预计预处理耗时: ~2h (单 CPU，可并行加速)

---

## 8. Motion Condition 训练采样分析 [v1.5 新增]

### 8.1 概述

Motion condition 采样是 HyMotion M2M 的核心能力之一。它决定了模型在推理时能处理多大范围的 condition pattern。当前有两个采样器版本:

| 采样器 | 使用场景 | 覆盖率 | 机制 |
|--------|---------|--------|------|
| **v2** | caption configs (`cond_mask_prob=0.1`) | ~40% | 两层混合: 60% 参数化 Tier-1 + 40% 模板 Tier-2 |
| **v3** | uncond configs (`cond_mask_prob=0.0`) | ~84% | Rank-K Boolean Tensor Prior, 数学统一 |

### 8.2 V3 Condition Sampler 详解

V3 sampler 是当前最先进的设计，核心是 **Rank-K Boolean Tensor Prior**:

```
M = ⊻_{k=1..K} (t_k ⊗ d_k)

其中:
  K ~ πK = (0.10, 0.55, 0.25, 0.07, 0.03)  对应 K ∈ {0,1,2,3,4}
  t_k ∈ {0,1}^T  — 时间模式 (从 6 种 temporal primitive 中采样)
  d_k ∈ {0,1}^198 — 维度模式 (从 5 种 dimensional kind 中采样)
```

#### 8.2.1 时间分布 (πT: 6 primitives)

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

#### 8.2.2 空间/维度分布 (πD: 5 kinds)

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
- ⚠️ **Gap**: `pos_only` 不包含 pelvis (joint 0)；`trans_only` 不控制非 root 的 position

#### 8.2.3 稀疏度控制

稀疏度由 **K (atom 数量)** 和 **temporal primitive 参数** 共同控制:

| K 值 | 概率 | 典型稀疏度 | 对应任务 |
|------|------|-----------|---------|
| K=0 | 10% | 100% 生成 (无条件) | T2M, unconditional |
| K=1 | 55% | 依 primitive: 1-100% | 大多数 M2M 任务 |
| K=2 | 25% | 多层条件叠加 | 复合约束 (如 trajectory + keyframe) |
| K=3 | 7% | 高密度条件 | 精细编辑/修复 |
| K=4 | 3% | 极高密度 | 接近完整约束 |

**关键**: K≥2 时，多个 atom 的 Boolean OR 产生更丰富的条件模式，这是 v3 优于 v2 的核心——v2 只能产生 v3 K=1 等价的模式子集。

#### 8.2.4 旋转 vs Position 模态覆盖

| 模态 | 支持方式 | 训练采样概率 |
|------|---------|-------------|
| **纯旋转条件** | `rot_only` kind (22%) | ~22% × (1-0.10) ≈ 20% |
| **纯位置条件** | `pos_only` kind (30%) | ~30% × (1-0.10) ≈ 27% |
| **纯轨迹条件** | `trans_only` kind (10%) | ~10% × (1-0.10) ≈ 9% |
| **混合模态** | `mixed` kind (18%) + K≥2 的 cross-kind | ~18% + 多 atom 组合 |
| **全模态** | `all_dim` kind (20%) | ~20% × (1-0.10) ≈ 18% |

### 8.3 V2 vs V3 对比与 Caption 配置的 Gap

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

### 8.4 当前方案的不足与改进方向

| 不足 | 严重度 | 改进方向 |
|------|--------|---------|
| Caption configs 仍用 v2 sampler (40% 覆盖) | P1 | 统一使用 v3 |
| 无显式 multi-segment temporal primitive | P2 | K≥2 的 interval OR 近似覆盖 |
| Pelvis position 不在 `pos_only` 覆盖中 | P2 | 已有 `trans_only` 覆盖 translation |
| 无 contact/foot height 条件类型 | P2 | CCFM (§6) 解决 |
| 无 velocity 条件类型 | P3 | 需要扩展 dimensional kind |

---

## 9. 实验计划（按优先级排序） [v1.5 重写]

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
transl_aug_prob = 0.5              # D3 修复: 启用 translation augmentation (仅版本 A)
timestep_squared_weighting = False  # 移除 t² weighting
aux_joint_pos_weight = 50.0
aux_joint_vel_weight = 500.0
aux_fk_consistency_weight = 1500.0
# position loss 统一在 relative-to-root 空间计算
```

#### 实验 E1: SMPL Root + Uncond (Baseline)

| 项目 | 值 |
|------|-----|
| **Root 表征** | 版本 A: SMPL Root (198-dim) |
| **训练任务** | Unconditional (cond_mask_prob=0.0) |
| **Condition Sampler** | v3 |
| **修改内容** | 启用 FK loss + transl_aug + 移除 t² + position loss relative-to-root |
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
| **修改内容** | 同 E1 + v2→v3 sampler 升级 + null_embedding_source 修复 |
| **Config 基础** | `hymotion_m2m_v2_caption_local_046b.py` |
| **GPU** | 48 × V100 |
| **预期效果** | Text conditioning 恢复 + 滑步改善 |
| **有效概率** | 85% |

#### 实验 E3: KIMODO Root + Uncond

| 项目 | 值 |
|------|-----|
| **Root 表征** | 版本 B: KIMODO Root (197-dim) |
| **训练任务** | Unconditional (cond_mask_prob=0.0) |
| **Condition Sampler** | v3 (适配 197-dim) |
| **修改内容** | 新 root 表征 + ADMM 预处理 + 新 mean/std |
| **Config 基础** | 新建 `hymotion_m2m_v2_kimodo_uncond_046b.py` |
| **GPU** | 48 × V100 |
| **预期效果** | 在 E1 基础上进一步减少滑步（ADMM smooth translation 效果） |
| **有效概率** | 70% (KIMODO 论文验证有效，但迁移到我们的 SMPL-22 表征可能有 gap) |

#### 实验 E4: KIMODO Root + Caption

| 项目 | 值 |
|------|-----|
| **Root 表征** | 版本 B: KIMODO Root (197-dim) |
| **训练任务** | Caption (cond_mask_prob=0.1) |
| **Condition Sampler** | v3 (适配 197-dim) |
| **修改内容** | 同 E3 + caption 支持 |
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

### 13.2 降级方案

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

## 14. 实施路线图 [v1.5 更新]

### Phase 0: Dual Root + Loss 对齐 + 首批实验 (Week 1-2) — v1.5 核心

```
✅ v2 caption bugs 修复 (2026-03-27 ~ 2026-05-12):
   - B1/B2-ext: bundle params 保存 + null embedding 加载链修复 (影响 v2 caption resume/phase/soar)
   - B5/B6: text token OOD + null embedding 分布 (影响 v2 caption)
   - 6 个 v2 caption 配置已添加 null_embedding_source (phase1/phase2/soar × local/global)
✅ v2 uncond_local: 无 bug 影响，无需修复
□ 实现 KIMODO Root 表征 (§7):
  □ 实现 smpl_root_to_kimodo_root() 和 kimodo_root_to_smpl_root() 转换函数
  □ 实现 ADMM XZ 平滑 (移植 KIMODO smooth_root.py)
  □ 数据预处理: 生成 197-dim KIMODO root 版本数据 + mean/std
  □ 单元测试: 可逆性验证 (SMPL → KIMODO → SMPL roundtrip)
□ Loss 对齐 (§7.4, §7.5):
  □ Position loss 改为 relative-to-root 空间
  □ 移除 t² timestep weighting (config 修改)
□ 滑步修复 — config-only (§1.2.3):
  □ 启用 FK keypoint loss: keypoints3d_weight=10.0 (D1)
  □ 启用 translation augmentation: transl_aug_prob=0.5 (D3)
□ Config 准备:
  □ E1: hymotion_m2m_v2_smpl_uncond_046b.py (版本 A)
  □ E2: hymotion_m2m_v2_smpl_caption_046b.py (版本 A + caption)
  □ E3: hymotion_m2m_v2_kimodo_uncond_046b.py (版本 B)
  □ E4: hymotion_m2m_v2_kimodo_caption_046b.py (版本 B + caption)
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

## 附录 D: KIMODO Root 转换规范

### D.1 维度映射

```
SMPL Root (135-dim) → KIMODO Root (134-dim) 转换:

SMPL:                          KIMODO:
[0:3]   abs_trans    ──┬──→   [0:3]   smooth_trans = ADMM(abs_trans.xz) + abs_trans.y
                       └──→   [131:134] trans_residual = abs_trans - smooth_trans
[3:9]   root_rot6d   ──→     [3:5]   heading = [cos(yaw), sin(yaw)]  (从 rot6d 提取 yaw)
[9:135]  body_rot     ──→     [5:131]  body_rot  (不变)

扩展版:
SMPL (198-dim) → KIMODO (197-dim):
[135:198] pos_rel_pelvis ──→  [134:197] pos_rel_smooth_root (参考系切换)
```

### D.2 ADMM 平滑参数

| 参数 | 值 | 说明 |
|------|-----|------|
| margin | 0.06m (6cm) | 平滑 XZ 轨迹的最大偏移约束 |
| step_size | 0.25 × sqrt(diag_max) | 自适应步长 |
| iterations | 100 per level | 每层优化迭代数 |
| smoothed_axes | XZ only | Y 轴高度保持原始值 |

### D.3 Heading 提取

```python
def extract_yaw(root_rotmat: Tensor) -> Tensor:
    """
    从 3×3 旋转矩阵提取 yaw (heading) 角度。
    
    假设旋转矩阵 R = Ry(yaw) @ Rx(pitch) @ Rz(roll)
    yaw = atan2(R[0,2], R[2,2])
    
    对于人体 pelvis，pitch/roll 通常 < 5°，yaw 占主导。
    """
    return torch.atan2(root_rotmat[..., 0, 2], root_rotmat[..., 2, 2])

def yaw_to_rot6d(yaw: Tensor) -> Tensor:
    """
    从 yaw 角度构建纯 yaw 旋转的 rot6d 表示。
    
    Ry(θ) = [[cos θ, 0, sin θ],
              [0,     1, 0    ],
              [-sin θ, 0, cos θ]]
    
    rot6d = [Ry[:, 0], Ry[:, 1]] = [cos θ, 0, -sin θ, 0, 1, 0]
    """
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    zeros = torch.zeros_like(yaw)
    ones = torch.ones_like(yaw)
    
    # rot6d = first two columns of rotation matrix, flattened
    return torch.stack([cos_y, zeros, -sin_y, zeros, ones, zeros], dim=-1)
```

### D.4 可逆性测试规范

```python
def test_roundtrip_conversion():
    """
    单元测试: SMPL → KIMODO → SMPL roundtrip 精度验证。
    
    验收标准:
    - Translation: max_error < 1e-5 (float32 精度)
    - Body rotation: exact (零误差)
    - Root rotation yaw: max_error < 1e-5
    - Root rotation pitch/roll: 丢失 (但 pelvis pitch/roll 通常 < 5°)
    """
    # 生成随机 SMPL motion
    smpl_motion = random_smpl_motion(B=4, T=196, dim=135)
    
    # Roundtrip
    kimodo_motion = smpl_root_to_kimodo_root(smpl_motion)
    recovered_smpl = kimodo_root_to_smpl_root(kimodo_motion)
    
    # 验证 translation
    assert (smpl_motion[..., 0:3] - recovered_smpl[..., 0:3]).abs().max() < 1e-5
    
    # 验证 body rotation (exact)
    assert (smpl_motion[..., 9:135] - recovered_smpl[..., 9:135]).abs().max() < 1e-7
    
    # 验证 root rotation yaw
    original_yaw = extract_yaw(rot6d_to_rotmat(smpl_motion[..., 3:9]))
    recovered_yaw = extract_yaw(rot6d_to_rotmat(recovered_smpl[..., 3:9]))
    assert (original_yaw - recovered_yaw).abs().max() < 1e-5
```

---

*文档结束。v1.5 核心更新: 双 Root 表征方案 (SMPL vs KIMODO) + Loss 对齐 + Motion Condition 采样分析 + 优先级实验计划 (E1-E4 首批 4×48 V100)。方案长期目标: 通过密度感知的条件解耦编排 (CDO-FM)，实现 text 和 motion condition 的自适应平衡，并以接触条件化 flow matching 提升生成质量。*
